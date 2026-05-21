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
}
