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

/// Infer a likely timestamp unit from absolute Unix timestamp magnitude.
///
/// This is only a migration helper. New code should prefer typed config or
/// dataset metadata instead of guessing.
pub fn infer_timestamp_unit(values: &[i64]) -> Option<TimestampUnit> {
    let sample = values.iter().copied().find(|value| *value != 0)?;
    let abs = sample.unsigned_abs();
    if abs >= 10_000_000_000_000_000 {
        Some(TimestampUnit::Nanoseconds)
    } else if abs >= 10_000_000_000_000 {
        Some(TimestampUnit::Microseconds)
    } else if abs >= 10_000_000_000 {
        Some(TimestampUnit::Milliseconds)
    } else {
        Some(TimestampUnit::Seconds)
    }
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

pub fn normalize_timestamps_to_millis(
    values: &[i64],
    unit: TimestampUnit,
) -> Result<Vec<i64>> {
    values
        .iter()
        .copied()
        .map(|value| timestamp_to_millis(value, unit))
        .collect()
}

pub fn normalize_timestamps_to_inferred_millis(values: &[i64]) -> Result<Vec<i64>> {
    let unit = infer_timestamp_unit(values).ok_or_else(|| anyhow::anyhow!("no timestamp values"))?;
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
        assert_eq!(timestamp_to_millis(1_700_000_000, TimestampUnit::Seconds)?, 1_700_000_000_000);
        assert_eq!(timestamp_to_millis(1_700_000_000_000, TimestampUnit::Milliseconds)?, 1_700_000_000_000);
        assert_eq!(timestamp_to_millis(1_700_000_000_000_000, TimestampUnit::Microseconds)?, 1_700_000_000_000);
        assert_eq!(timestamp_to_millis(1_700_000_000_000_000_000, TimestampUnit::Nanoseconds)?, 1_700_000_000_000);
        Ok(())
    }

    #[test]
    fn rejects_non_monotonic_timestamps() {
        assert!(validate_monotonic_timestamps(&[1, 2, 2, 3]).is_ok());
        assert!(validate_monotonic_timestamps(&[1, 3, 2]).is_err());
    }
}
