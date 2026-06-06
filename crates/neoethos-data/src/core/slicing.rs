//! Shared slicing helpers for the [`Ohlcv`] type.
//!
//! Phase 68 extraction: previously two near-duplicate `slice_ohlcv`
//! helpers lived in `neoethos-search::discovery` and
//! `neoethos-search::genetic::regime_labels`. The latter accepted an
//! optional fallback-timestamp slice for callers whose `Ohlcv` carried
//! `timestamp: None`. This module unifies both shapes behind a single
//! `Option<&[i64]>` parameter.

use crate::Ohlcv;

/// Slice an [`Ohlcv`] into the half-open range `[start_idx, end_idx)`.
///
/// When `ohlcv.timestamp` is `Some`, that vector is sliced and returned
/// inside `Some`. When it is `None` and `fallback_timestamps` is
/// `Some`, the fallback slice fills the timestamp field. When both are
/// absent, the returned `Ohlcv` carries `timestamp: None`.
///
/// Panics if `end_idx` exceeds any of the inner vectors' lengths.
pub fn slice_ohlcv(
    ohlcv: &Ohlcv,
    start_idx: usize,
    end_idx: usize,
    fallback_timestamps: Option<&[i64]>,
) -> Ohlcv {
    let timestamp = match (ohlcv.timestamp.as_ref(), fallback_timestamps) {
        (Some(ts), _) => Some(ts[start_idx..end_idx].to_vec()),
        (None, Some(fallback)) => Some(fallback[start_idx..end_idx].to_vec()),
        (None, None) => None,
    };
    Ohlcv {
        timestamp,
        open: ohlcv.open[start_idx..end_idx].to_vec(),
        high: ohlcv.high[start_idx..end_idx].to_vec(),
        low: ohlcv.low[start_idx..end_idx].to_vec(),
        close: ohlcv.close[start_idx..end_idx].to_vec(),
        volume: ohlcv
            .volume
            .as_ref()
            .map(|vol| vol[start_idx..end_idx].to_vec()),
    }
}

/// Slice an [`Ohlcv`] by the half-open timestamp range `[from_ms, to_ms)`
/// (epoch milliseconds, UTC). Returns the filtered subset together with
/// `(first_ms, last_ms)` of the kept span when at least one row survives.
///
/// The timestamp column MUST be present and sorted ascending — both
/// guaranteed by the canonical Vortex loader (`vortex_array_to_ohlcv`
/// normalises to ms and `normalize_ohlcv` sorts on write). We locate the
/// boundary indices with `partition_point` (binary search, O(log n)):
///   - `start` = first index whose `ts >= from_ms`
///   - `end`   = first index whose `ts >= to_ms`   (exclusive upper bound)
/// then defer to [`slice_ohlcv`] for the actual column slicing so the two
/// share one code path.
///
/// Returns `Err` if the timestamp column is absent (the date filter is
/// meaningless without it). An empty result (`start == end`) is NOT an
/// error here — the caller decides whether zero kept rows is fatal, so
/// the "0 rows in range" failure can carry the requested dates in its
/// message.
pub fn slice_ohlcv_by_date_range_ms(
    ohlcv: &Ohlcv,
    from_ms: i64,
    to_ms: i64,
) -> Result<(Ohlcv, Option<(i64, i64)>), String> {
    let ts = ohlcv
        .timestamp
        .as_ref()
        .ok_or_else(|| "dataset has no timestamp column — cannot slice by date".to_string())?;

    // Half-open [from_ms, to_ms): start = first ts >= from_ms,
    // end = first ts >= to_ms. partition_point requires the predicate to
    // be monotone over the slice (true... then false...), which holds for
    // an ascending timestamp vector.
    let start = ts.partition_point(|&t| t < from_ms);
    let end = ts.partition_point(|&t| t < to_ms);

    if start >= end {
        // Empty result — return an empty Ohlcv of the same shape, no span.
        return Ok((slice_ohlcv(ohlcv, start, start, None), None));
    }

    let sliced = slice_ohlcv(ohlcv, start, end, None);
    let span = Some((ts[start], ts[end - 1]));
    Ok((sliced, span))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlcv() -> Ohlcv {
        Ohlcv {
            timestamp: Some(vec![100, 200, 300, 400]),
            open: vec![1.0, 2.0, 3.0, 4.0],
            high: vec![1.1, 2.1, 3.1, 4.1],
            low: vec![0.9, 1.9, 2.9, 3.9],
            close: vec![1.05, 2.05, 3.05, 4.05],
            volume: Some(vec![10.0, 20.0, 30.0, 40.0]),
        }
    }

    #[test]
    fn slice_keeps_timestamp_when_present() {
        let sliced = slice_ohlcv(&sample_ohlcv(), 1, 3, None);
        assert_eq!(sliced.timestamp.as_deref(), Some(&[200_i64, 300][..]));
        assert_eq!(sliced.open, vec![2.0, 3.0]);
        assert_eq!(sliced.volume.as_deref(), Some(&[20.0_f64, 30.0][..]));
    }

    #[test]
    fn slice_uses_fallback_when_timestamp_is_none() {
        let mut ohlcv = sample_ohlcv();
        ohlcv.timestamp = None;
        let fallback = vec![10, 20, 30, 40];
        let sliced = slice_ohlcv(&ohlcv, 0, 2, Some(&fallback));
        assert_eq!(sliced.timestamp.as_deref(), Some(&[10_i64, 20][..]));
    }

    #[test]
    fn slice_returns_none_timestamp_when_both_absent() {
        let mut ohlcv = sample_ohlcv();
        ohlcv.timestamp = None;
        let sliced = slice_ohlcv(&ohlcv, 0, 2, None);
        assert!(sliced.timestamp.is_none());
    }

    #[test]
    fn date_range_half_open_picks_correct_rows() {
        // Bars at ms 100, 200, 300, 400. Range [200, 400) keeps 200, 300.
        let (sliced, span) = slice_ohlcv_by_date_range_ms(&sample_ohlcv(), 200, 400).unwrap();
        assert_eq!(sliced.timestamp.as_deref(), Some(&[200_i64, 300][..]));
        assert_eq!(sliced.open, vec![2.0, 3.0]);
        assert_eq!(span, Some((200, 300)));
    }

    #[test]
    fn date_range_empty_when_no_rows_in_window() {
        // Range entirely after the data → 0 kept rows, no span.
        let (sliced, span) = slice_ohlcv_by_date_range_ms(&sample_ohlcv(), 1000, 2000).unwrap();
        assert!(sliced.timestamp.as_deref().is_some_and(|t| t.is_empty()));
        assert!(span.is_none());
    }

    #[test]
    fn date_range_errors_without_timestamps() {
        let mut ohlcv = sample_ohlcv();
        ohlcv.timestamp = None;
        let err = slice_ohlcv_by_date_range_ms(&ohlcv, 0, 100).unwrap_err();
        assert!(err.contains("timestamp"), "unexpected error: {err}");
    }

    /// Format-compatibility gate: build a tiny dataset with known
    /// timestamps, write it to a temp `<root>/symbol=/timeframe=/data.vortex`,
    /// slice it by a DATE range, write the slice to a second temp root via
    /// the SAME Vortex writer discover uses, then RELOAD the slice via the
    /// SAME reader discover uses (`load_symbol_timeframe`) and assert the
    /// kept rows + span survive the round-trip. This is the gate that
    /// proves `slice-dataset` output is byte-compatible with discovery.
    #[test]
    fn slice_dataset_round_trips_through_vortex_io() {
        use crate::{Ohlcv, load_symbol_timeframe, write_symbol_timeframe_vortex};
        use std::time::{SystemTime, UNIX_EPOCH};

        // 10 daily bars, 2020-01-01 00:00 UTC onward (1 day = 86_400_000 ms).
        let day_ms = 86_400_000_i64;
        let base = 1_577_836_800_000_i64; // 2020-01-01T00:00:00Z
        let n = 10usize;
        let mut timestamp = Vec::with_capacity(n);
        let (mut open, mut high, mut low, mut close, mut volume) =
            (vec![], vec![], vec![], vec![], vec![]);
        for i in 0..n {
            let p = 1.0 + (i as f64) * 0.001;
            timestamp.push(base + (i as i64) * day_ms);
            open.push(p);
            high.push(p + 0.002);
            low.push(p - 0.002);
            close.push(p + 0.001);
            volume.push(100.0 + i as f64);
        }
        let full = Ohlcv {
            timestamp: Some(timestamp.clone()),
            open,
            high,
            low,
            close,
            volume: Some(volume),
        };

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp = std::env::temp_dir().join(format!(
            "neoethos_slice_roundtrip_{}_{}",
            std::process::id(),
            nonce
        ));
        let src_root = tmp.join("src");
        let dst_root = tmp.join("dst");

        // Write the full source through the canonical writer, then reload it
        // the way the CLI command will (so the slicer sees ms-normalised ts).
        write_symbol_timeframe_vortex(&src_root, "EURUSD", "D1", &full)
            .expect("write source vortex");
        let reloaded_src =
            load_symbol_timeframe(&src_root, "EURUSD", "D1").expect("reload source vortex");
        assert_eq!(reloaded_src.len(), n, "source round-trip row count");

        // Keep [2020-01-03, 2020-01-07) → bars at day index 2,3,4,5 (4 rows).
        let from_ms = base + 2 * day_ms;
        let to_ms = base + 6 * day_ms;
        let (slice, span) =
            slice_ohlcv_by_date_range_ms(&reloaded_src, from_ms, to_ms).expect("slice by date");
        assert_eq!(slice.len(), 4, "expected 4 kept rows");
        assert_eq!(span, Some((base + 2 * day_ms, base + 5 * day_ms)));

        // Persist the slice and reload it via the SAME discover-facing reader.
        write_symbol_timeframe_vortex(&dst_root, "EURUSD", "D1", &slice)
            .expect("write sliced vortex");
        let reloaded_slice =
            load_symbol_timeframe(&dst_root, "EURUSD", "D1").expect("reload sliced vortex");

        assert_eq!(reloaded_slice.len(), 4, "reloaded slice row count");
        let ts = reloaded_slice
            .timestamp
            .as_ref()
            .expect("slice has timestamps");
        assert_eq!(ts.first().copied(), Some(base + 2 * day_ms));
        assert_eq!(ts.last().copied(), Some(base + 5 * day_ms));
        // Volume column survived the slice + round-trip.
        assert_eq!(
            reloaded_slice.volume.as_deref(),
            Some(&[102.0_f64, 103.0, 104.0, 105.0][..])
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
