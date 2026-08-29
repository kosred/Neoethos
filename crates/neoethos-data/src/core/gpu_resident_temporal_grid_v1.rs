//! Dependency-free temporal-grid arithmetic shared by resident Quant-v3 and
//! Session-v2 owner preflight.
//!
//! This module does not infer a timeframe from observed gaps. The caller must
//! supply the canonical fixed timeframe, and every observed open must remain
//! on that epoch grid. Weekend/market gaps are legal only when they are exact
//! positive multiples of the declared duration.

use std::error::Error as StdError;
use std::fmt;

pub(crate) const UTC_DAY_MILLIS_V2: i64 = 86_400_000;
pub(crate) const ASIAN_SESSION_MILLIS_V2: i64 = 8 * 60 * 60 * 1_000;
pub(crate) const TRADING_SESSIONS_PER_YEAR_V3: u64 = 252;
pub(crate) const TRADING_DAYS_PER_WEEK_V3: u64 = 5;
pub(crate) const MINIMUM_ASIAN_ORB_BARS_V3: u64 = 12;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ResidentTemporalGridErrorV1 {
    EmptyTimestamps,
    InvalidTimeframe(i64),
    TimeframeDoesNotDivideUtcDay(i64),
    TimeframeDoesNotDivideAsianSession(i64),
    InsufficientAsianSessionBars(u64),
    TimestampOffGrid { row: usize, timestamp_ms: i64 },
    TimestampGapInvalid { row: usize, gap_ms: i64 },
    ArithmeticOverflow(&'static str),
}

impl fmt::Display for ResidentTemporalGridErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTimestamps => formatter.write_str("resident temporal grid is empty"),
            Self::InvalidTimeframe(value) => {
                write!(
                    formatter,
                    "resident timeframe_millis must be positive, got {value}"
                )
            }
            Self::TimeframeDoesNotDivideUtcDay(value) => write!(
                formatter,
                "resident timeframe {value} ms does not divide the fixed UTC day"
            ),
            Self::TimeframeDoesNotDivideAsianSession(value) => write!(
                formatter,
                "resident timeframe {value} ms does not divide the 00:00-08:00 UTC Asian session"
            ),
            Self::InsufficientAsianSessionBars(actual) => write!(
                formatter,
                "resident Quant-v3 requires at least twelve bars in the Asian session, got {actual}"
            ),
            Self::TimestampOffGrid { row, timestamp_ms } => write!(
                formatter,
                "resident timestamp row {row} ({timestamp_ms}) is off the declared epoch grid"
            ),
            Self::TimestampGapInvalid { row, gap_ms } => write!(
                formatter,
                "resident timestamp gap ending at row {row} is not a positive grid multiple: {gap_ms}"
            ),
            Self::ArithmeticOverflow(field) => {
                write!(formatter, "resident temporal {field} overflowed")
            }
        }
    }
}

impl StdError for ResidentTemporalGridErrorV1 {}

/// Exact fixed-intraday facts consumed by Quant-v3 route construction.
/// Private fields prevent a caller from claiming a checked grid by assembling
/// a DTO. This receipt is intentionally move-only.
#[must_use = "the admitted temporal grid must move into Quant-v3 producer preflight"]
#[derive(Debug)]
pub(crate) struct AdmittedFixedIntradayGridV1 {
    timeframe_millis: u64,
    bars_per_asian_session: u64,
    bars_per_utc_day: u64,
    bars_per_trading_week: u64,
    annualization_periods_per_year: u64,
}

impl AdmittedFixedIntradayGridV1 {
    pub(crate) const fn timeframe_millis(&self) -> u64 {
        self.timeframe_millis
    }

    pub(crate) const fn bars_per_asian_session(&self) -> u64 {
        self.bars_per_asian_session
    }

    pub(crate) const fn bars_per_utc_day(&self) -> u64 {
        self.bars_per_utc_day
    }

    pub(crate) const fn bars_per_trading_week(&self) -> u64 {
        self.bars_per_trading_week
    }

    pub(crate) const fn annualization_periods_per_year(&self) -> u64 {
        self.annualization_periods_per_year
    }
}

pub(crate) fn admit_fixed_intraday_grid_v1(
    timeframe_millis: i64,
    timestamps: &[i64],
) -> Result<AdmittedFixedIntradayGridV1, ResidentTemporalGridErrorV1> {
    if timestamps.is_empty() {
        return Err(ResidentTemporalGridErrorV1::EmptyTimestamps);
    }
    if timeframe_millis <= 0 {
        return Err(ResidentTemporalGridErrorV1::InvalidTimeframe(
            timeframe_millis,
        ));
    }
    if UTC_DAY_MILLIS_V2.rem_euclid(timeframe_millis) != 0 {
        return Err(ResidentTemporalGridErrorV1::TimeframeDoesNotDivideUtcDay(
            timeframe_millis,
        ));
    }
    if ASIAN_SESSION_MILLIS_V2.rem_euclid(timeframe_millis) != 0 {
        return Err(
            ResidentTemporalGridErrorV1::TimeframeDoesNotDivideAsianSession(timeframe_millis),
        );
    }
    let bars_per_asian_session = u64::try_from(ASIAN_SESSION_MILLIS_V2 / timeframe_millis)
        .map_err(|_| ResidentTemporalGridErrorV1::ArithmeticOverflow("Asian-session bars"))?;
    if bars_per_asian_session < MINIMUM_ASIAN_ORB_BARS_V3 {
        return Err(ResidentTemporalGridErrorV1::InsufficientAsianSessionBars(
            bars_per_asian_session,
        ));
    }
    for (row, &timestamp) in timestamps.iter().enumerate() {
        if timestamp.rem_euclid(timeframe_millis) != 0 {
            return Err(ResidentTemporalGridErrorV1::TimestampOffGrid {
                row,
                timestamp_ms: timestamp,
            });
        }
    }
    for (offset, pair) in timestamps.windows(2).enumerate() {
        let gap =
            pair[1]
                .checked_sub(pair[0])
                .ok_or(ResidentTemporalGridErrorV1::ArithmeticOverflow(
                    "timestamp gap",
                ))?;
        if gap <= 0 || gap.rem_euclid(timeframe_millis) != 0 {
            return Err(ResidentTemporalGridErrorV1::TimestampGapInvalid {
                row: offset + 1,
                gap_ms: gap,
            });
        }
    }
    let bars_per_utc_day = u64::try_from(UTC_DAY_MILLIS_V2 / timeframe_millis)
        .map_err(|_| ResidentTemporalGridErrorV1::ArithmeticOverflow("UTC-day bars"))?;
    let bars_per_trading_week = bars_per_utc_day
        .checked_mul(TRADING_DAYS_PER_WEEK_V3)
        .ok_or(ResidentTemporalGridErrorV1::ArithmeticOverflow(
            "trading-week bars",
        ))?;
    let annualization_periods_per_year = bars_per_utc_day
        .checked_mul(TRADING_SESSIONS_PER_YEAR_V3)
        .ok_or(ResidentTemporalGridErrorV1::ArithmeticOverflow(
            "annualization periods",
        ))?;
    Ok(AdmittedFixedIntradayGridV1 {
        timeframe_millis: u64::try_from(timeframe_millis)
            .map_err(|_| ResidentTemporalGridErrorV1::InvalidTimeframe(timeframe_millis))?,
        bars_per_asian_session,
        bars_per_utc_day,
        bars_per_trading_week,
        annualization_periods_per_year,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(step_ms: i64, rows: usize) -> Vec<i64> {
        let start = 1_700_006_400_000_i64;
        (0..rows)
            .map(|row| start + i64::try_from(row).expect("small row") * step_ms)
            .collect()
    }

    #[test]
    fn m30_closes_the_orb_floor_and_derives_252_session_annualization() {
        let admitted = admit_fixed_intraday_grid_v1(30 * 60_000, &grid(30 * 60_000, 20))
            .expect("M30 fixed grid");
        assert_eq!(admitted.bars_per_asian_session(), 16);
        assert_eq!(admitted.bars_per_utc_day(), 48);
        assert_eq!(admitted.bars_per_trading_week(), 240);
        assert_eq!(admitted.annualization_periods_per_year(), 48 * 252);
    }

    #[test]
    fn h1_and_calendar_or_off_grid_inputs_fail_closed() {
        assert!(matches!(
            admit_fixed_intraday_grid_v1(60 * 60_000, &grid(60 * 60_000, 20)),
            Err(ResidentTemporalGridErrorV1::InsufficientAsianSessionBars(8))
        ));
        assert!(matches!(
            admit_fixed_intraday_grid_v1(86_400_000, &grid(86_400_000, 20)),
            Err(ResidentTemporalGridErrorV1::InsufficientAsianSessionBars(_))
                | Err(ResidentTemporalGridErrorV1::TimeframeDoesNotDivideAsianSession(_))
        ));
        let mut off_grid = grid(60_000, 4);
        off_grid[2] += 1;
        assert!(matches!(
            admit_fixed_intraday_grid_v1(60_000, &off_grid),
            Err(ResidentTemporalGridErrorV1::TimestampOffGrid { row: 2, .. })
        ));
    }

    #[test]
    fn exact_grid_multiple_market_gaps_are_allowed_but_partial_gaps_are_not() {
        let mut with_weekend_gap = grid(60_000, 4);
        with_weekend_gap[2] += 2 * UTC_DAY_MILLIS_V2;
        with_weekend_gap[3] += 2 * UTC_DAY_MILLIS_V2;
        let _admitted = admit_fixed_intraday_grid_v1(60_000, &with_weekend_gap)
            .expect("whole-period weekend gap");

        let mut partial = grid(60_000, 4);
        partial[2] += 30_000;
        assert!(admit_fixed_intraday_grid_v1(60_000, &partial).is_err());
    }
}
