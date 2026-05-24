use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Explicit timestamp unit used by OHLCV, feature, and evaluation paths.
///
/// The codebase historically mixed names such as `*_ns` with evaluation logic
/// that expects milliseconds. Keeping the unit explicit is the first step to
/// making feature generation, MTF alignment, session features, backtests, and
/// CPU/GPU evaluators share the same time contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimestampUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

impl TimestampUnit {
    pub fn scale_to_millis(self) -> i64 {
        match self {
            TimestampUnit::Seconds => 1_000,
            TimestampUnit::Milliseconds => 1,
            TimestampUnit::Microseconds => 1_000,
            TimestampUnit::Nanoseconds => 1_000_000,
        }
    }

    pub fn scale_from_millis(self) -> i64 {
        match self {
            TimestampUnit::Seconds => 1_000,
            TimestampUnit::Milliseconds => 1,
            TimestampUnit::Microseconds => 1_000,
            TimestampUnit::Nanoseconds => 1_000_000,
        }
    }
}

fn classify_magnitude(value: i64) -> TimestampUnit {
    let abs = value.unsigned_abs();
    if abs >= 10_000_000_000_000_000 {
        TimestampUnit::Nanoseconds
    } else if abs >= 10_000_000_000_000 {
        TimestampUnit::Microseconds
    } else if abs >= 10_000_000_000 {
        TimestampUnit::Milliseconds
    } else {
        TimestampUnit::Seconds
    }
}

/// Infer a likely timestamp unit from absolute Unix timestamp magnitude.
///
/// #156: this is only a migration helper. New code should prefer typed
/// config or dataset metadata instead of guessing. To make the guess robust
/// against single-row upstream corruption (e.g. one stray row with
/// `timestamp = 1`), we sample up to the first 16 non-zero values and use
/// the **most common** magnitude bucket. If we see a heterogeneous sample
/// (no bucket gets ≥75% of the votes) we conservatively return `None` so
/// the caller bails rather than silently mis-folding the dataset.
pub fn infer_timestamp_unit(values: &[i64]) -> Option<TimestampUnit> {
    let sample: Vec<TimestampUnit> = values
        .iter()
        .copied()
        .filter(|value| *value != 0)
        .take(16)
        .map(classify_magnitude)
        .collect();
    if sample.is_empty() {
        return None;
    }
    let mut votes = [0usize; 4];
    for &u in &sample {
        let idx = match u {
            TimestampUnit::Seconds => 0,
            TimestampUnit::Milliseconds => 1,
            TimestampUnit::Microseconds => 2,
            TimestampUnit::Nanoseconds => 3,
        };
        votes[idx] += 1;
    }
    let (best_idx, &best_count) = votes.iter().enumerate().max_by_key(|(_, count)| *count)?;
    // ≥75% threshold catches both "all 16 agree" (normal case) and the
    // realistic mixed-bag of 1 corrupted row among 16 good ones (15/16
    // = 93.75%, well over the threshold). 13/16 = 81.25% still passes;
    // 12/16 = 75% exactly passes. A 50/50 split fails — that points at
    // a real schema problem the caller should surface.
    if (best_count as f64) / (sample.len() as f64) < 0.75 {
        return None;
    }
    Some(match best_idx {
        0 => TimestampUnit::Seconds,
        1 => TimestampUnit::Milliseconds,
        2 => TimestampUnit::Microseconds,
        3 => TimestampUnit::Nanoseconds,
        _ => unreachable!("votes array has exactly 4 buckets"),
    })
}

pub fn timestamp_to_millis(value: i64, unit: TimestampUnit) -> Result<i64> {
    match unit {
        TimestampUnit::Seconds => value
            .checked_mul(1_000)
            .ok_or_else(|| anyhow::anyhow!("timestamp seconds->millis overflow")),
        TimestampUnit::Milliseconds => Ok(value),
        TimestampUnit::Microseconds => Ok(value / 1_000),
        TimestampUnit::Nanoseconds => Ok(value / 1_000_000),
    }
}

pub fn timestamp_from_millis(value: i64, unit: TimestampUnit) -> Result<i64> {
    match unit {
        TimestampUnit::Seconds => Ok(value / 1_000),
        TimestampUnit::Milliseconds => Ok(value),
        TimestampUnit::Microseconds => value
            .checked_mul(1_000)
            .ok_or_else(|| anyhow::anyhow!("timestamp millis->micros overflow")),
        TimestampUnit::Nanoseconds => value
            .checked_mul(1_000_000)
            .ok_or_else(|| anyhow::anyhow!("timestamp millis->nanos overflow")),
    }
}

pub fn normalize_timestamps_to_millis(values: &[i64], unit: TimestampUnit) -> Result<Vec<i64>> {
    values
        .iter()
        .copied()
        .map(|value| timestamp_to_millis(value, unit))
        .collect()
}

pub fn normalize_timestamps_to_inferred_millis(values: &[i64]) -> Result<Vec<i64>> {
    let unit =
        infer_timestamp_unit(values).ok_or_else(|| anyhow::anyhow!("no timestamp values"))?;
    normalize_timestamps_to_millis(values, unit)
}

pub fn validate_monotonic_timestamps(values: &[i64]) -> Result<()> {
    for window in values.windows(2) {
        if window[1] < window[0] {
            bail!(
                "timestamps must be sorted ascending: {} came after {}",
                window[1],
                window[0]
            );
        }
    }
    Ok(())
}

pub fn day_key_from_millis(timestamp_ms: i64) -> i64 {
    timestamp_ms.div_euclid(86_400_000)
}

pub fn month_key_from_millis(timestamp_ms: i64) -> i64 {
    // Stable monotonic month-ish key used only for bucketing when calendar
    // decoding is not required. Calendar-aware reporting should use chrono.
    timestamp_ms.div_euclid(86_400_000 * 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_timestamp_unit_from_epoch_magnitude() {
        assert_eq!(
            infer_timestamp_unit(&[1_700_000_000]),
            Some(TimestampUnit::Seconds)
        );
        assert_eq!(
            infer_timestamp_unit(&[1_700_000_000_000]),
            Some(TimestampUnit::Milliseconds)
        );
        assert_eq!(
            infer_timestamp_unit(&[1_700_000_000_000_000]),
            Some(TimestampUnit::Microseconds)
        );
        assert_eq!(
            infer_timestamp_unit(&[1_700_000_000_000_000_000]),
            Some(TimestampUnit::Nanoseconds)
        );
    }

    #[test]
    fn converts_supported_units_to_millis() -> Result<()> {
        assert_eq!(
            timestamp_to_millis(1_700_000_000, TimestampUnit::Seconds)?,
            1_700_000_000_000
        );
        assert_eq!(
            timestamp_to_millis(1_700_000_000_000, TimestampUnit::Milliseconds)?,
            1_700_000_000_000
        );
        assert_eq!(
            timestamp_to_millis(1_700_000_000_000_000, TimestampUnit::Microseconds)?,
            1_700_000_000_000
        );
        assert_eq!(
            timestamp_to_millis(1_700_000_000_000_000_000, TimestampUnit::Nanoseconds)?,
            1_700_000_000_000
        );
        Ok(())
    }

    #[test]
    fn rejects_non_monotonic_timestamps() {
        assert!(validate_monotonic_timestamps(&[1, 2, 2, 3]).is_ok());
        assert!(validate_monotonic_timestamps(&[1, 3, 2]).is_err());
    }

    #[test]
    fn infer_timestamp_unit_tolerates_single_corrupt_row() {
        // 1 corrupted row + 15 legitimate millis rows → still infers ms.
        let mut values = vec![1_i64]; // corrupt: classifies as Seconds.
        for i in 0..15 {
            values.push(1_700_000_000_000 + i);
        }
        assert_eq!(
            infer_timestamp_unit(&values),
            Some(TimestampUnit::Milliseconds)
        );
    }

    #[test]
    fn infer_timestamp_unit_refuses_heterogeneous_sample() {
        // 8 seconds + 8 millis → no bucket reaches 75%, refuses to guess.
        let mut values = vec![];
        for i in 0..8 {
            values.push(1_700_000_000 + i);
        }
        for i in 0..8 {
            values.push(1_700_000_000_000 + i);
        }
        assert_eq!(infer_timestamp_unit(&values), None);
    }
}
