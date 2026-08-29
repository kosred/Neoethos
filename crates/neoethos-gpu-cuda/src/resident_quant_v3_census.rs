//! Frozen route/formula/validity census for resident Quant semantic-v3.
//!
//! Thirty-one routes retain the current `compute_quant_feature_columns_f64`
//! value bits and logical validity. Ten formerly unchanged routes and all
//! eight annualized-volatility routes migrate to the same Sun/OpenLibm exact
//! log graph used by CUDA. Fourteen temporal/session routes are also explicit
//! semantic-v3 migrations. Quant-v3 as a whole is never bitwise-v2 parity.

#![allow(dead_code)]

pub const RESIDENT_QUANT_SEMANTIC_VERSION_V3: u32 = 3;
pub const RESIDENT_QUANT_FEATURE_COLUMN_COUNT_V3: usize = 63;
pub const RESIDENT_QUANT_V2_BITWISE_PRESERVED_ROUTE_COUNT_V3: usize = 31;
pub const RESIDENT_QUANT_V3_EXACT_LOG_MIGRATION_ROUTE_COUNT_V3: usize = 10;
pub const RESIDENT_QUANT_V3_ANNUALIZED_EXACT_LOG_MIGRATION_ROUTE_COUNT_V3: usize = 8;
pub const RESIDENT_QUANT_V3_TEMPORAL_MIGRATION_ROUTE_COUNT_V3: usize = 14;
pub const RESIDENT_QUANT_V3_EXACT_LOG_AFFECTED_ROUTE_COUNT_V3: usize = 18;
pub const RESIDENT_QUANT_V3_MIGRATED_ROUTE_COUNT_V3: usize = 32;
pub const RESIDENT_QUANT_TRADING_SESSIONS_PER_YEAR_V3: u64 = 252;
pub const RESIDENT_QUANT_CANONICAL_NAN_BITS_V3: u64 = 0x7ff8_0000_0000_0000;

pub const RESIDENT_QUANT_IMPLEMENTATION_ID_V3: &str = "neoethos.cuda.resident-quant.semantic-v3";
pub const RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V3: &str = "neoethos.quant.cpu-cuda.semantic-v3;sun-fdlibm-openlibm-e_log;commit=82e90aef0657289192efe77be89791c07dea0775;source-sha256=8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD;cpu-cuda-bit-tolerance=zero;real-log-accuracy=bounded-faithful-max-1ulp-reviewed-wide-domain;f64-fixed-order;canonical-ms-fixed-intraday;utc-day-open=00:00;asian-session=00:00-08:00;trading-sessions-per-year=252;annualization=sqrt(252*bars-per-utc-day);orb=asian-session-reset;validity=logical-u8-v3;fmad=false;ftz=false;prec-div=true;prec-sqrt=true";
pub const RESIDENT_QUANT_V2_TO_V3_MIGRATION_POLICY: &str = "neoethos.quant.migration.v2-to-v3;bitwise-preserved-v2-routes=31;migrated-existing-exact-log-routes=10;migrated-annualized-exact-log-routes=8;migrated-temporal-routes=14;changed-routes=32;trading_sessions_per_year=252;v2-artifacts=fail-closed;unversioned-artifacts=fail-closed;never-label-as-bitwise-v2-parity";
pub const RESIDENT_QUANT_OPERATION_SCHEDULE_V3: &str = "neoethos.quant.semantic-v3.single-thread-fixed-order-linear-scan;one-native-launch;fixed-maximum-lookback=500-bars;utc-day-week-asian-state=O(1)-per-row;no-feature-d2h";

pub const RESIDENT_QUANT_COLUMN_NAMES_V3: [&str; 63] = [
    "quant_close",
    "quant_return_1",
    "quant_return_2",
    "quant_return_3",
    "quant_return_5",
    "quant_return_8",
    "quant_return_13",
    "quant_return_21",
    "quant_log_return",
    "quant_log_volatility",
    "quant_realized_vol_5",
    "quant_realized_vol_10",
    "quant_realized_vol_20",
    "quant_realized_vol_50",
    "quant_gk_vol_10",
    "quant_gk_vol_20",
    "quant_parkinson_vol_10",
    "quant_parkinson_vol_20",
    "quant_vol_ratio",
    "quant_hurst_100",
    "quant_autocorr_1",
    "quant_autocorr_5",
    "quant_autocorr_10",
    "quant_efficiency_ratio_10",
    "quant_efficiency_ratio_20",
    "quant_skewness_30",
    "quant_kurtosis_30",
    "quant_kyle_lambda",
    "quant_vpin",
    "quant_amihud_illiquidity",
    "quant_roll_spread",
    "quant_consec_up",
    "quant_consec_down",
    "quant_inside_bar",
    "quant_outside_bar",
    "quant_body_ratio",
    "quant_upper_shadow",
    "quant_lower_shadow",
    "quant_prev_day_h_dist",
    "quant_prev_day_l_dist",
    "quant_prev_week_h_dist",
    "quant_prev_week_l_dist",
    "quant_orb_4",
    "quant_orb_8",
    "quant_orb_12",
    "quant_amd_phase",
    "quant_wyckoff",
    "quant_engulfing_vol",
    "quant_pivot_dist",
    "quant_r1_dist",
    "quant_r2_dist",
    "quant_s1_dist",
    "quant_s2_dist",
    "quant_cam_r3_dist",
    "quant_cam_s3_dist",
    "quant_zscore_20",
    "quant_zscore_50",
    "quant_fractal_dim",
    "quant_rvol_10",
    "quant_rvol_20",
    "quant_rvol_50",
    "quant_delta_volume",
    "quant_cum_delta_zscore",
];

/// Exact migration census in canonical schema order. These are the only
/// routes whose value or previously-missing temporal semantics change in v3.
pub const RESIDENT_QUANT_V3_CHANGED_COLUMN_NAMES_V3: [&str; 32] = [
    "quant_log_return",
    "quant_log_volatility",
    "quant_realized_vol_5",
    "quant_realized_vol_10",
    "quant_realized_vol_20",
    "quant_realized_vol_50",
    "quant_gk_vol_10",
    "quant_gk_vol_20",
    "quant_parkinson_vol_10",
    "quant_parkinson_vol_20",
    "quant_vol_ratio",
    "quant_hurst_100",
    "quant_autocorr_1",
    "quant_autocorr_5",
    "quant_autocorr_10",
    "quant_skewness_30",
    "quant_kurtosis_30",
    "quant_prev_day_h_dist",
    "quant_prev_day_l_dist",
    "quant_prev_week_h_dist",
    "quant_prev_week_l_dist",
    "quant_orb_4",
    "quant_orb_8",
    "quant_orb_12",
    "quant_pivot_dist",
    "quant_r1_dist",
    "quant_r2_dist",
    "quant_s1_dist",
    "quant_s2_dist",
    "quant_cam_r3_dist",
    "quant_cam_s3_dist",
    "quant_fractal_dim",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentQuantRouteLineageV3 {
    V2BitwisePreserved,
    V3ExactLogMigration,
    V3AnnualizedExactLogMigration,
    V3TemporalSessionMigration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentQuantValidityRuleV3 {
    AlwaysValid,
    FixedWarmup,
    FixedWarmupOrZeroDenominator,
    PreviousUtcDayBoundaryOrZeroDenominator,
    PreviousFiveTradingDaysOrZeroDenominator,
    AsianOpeningRangeObservedBars,
    CumulativeDeltaPrefixOrZeroDenominator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentQuantRouteCensusV3 {
    pub name: &'static str,
    pub formula_id: &'static str,
    pub fixed_warmup_bars: u16,
    pub lineage: ResidentQuantRouteLineageV3,
    pub validity: ResidentQuantValidityRuleV3,
}

const fn route(
    name: &'static str,
    formula_id: &'static str,
    fixed_warmup_bars: u16,
    lineage: ResidentQuantRouteLineageV3,
    validity: ResidentQuantValidityRuleV3,
) -> ResidentQuantRouteCensusV3 {
    ResidentQuantRouteCensusV3 {
        name,
        formula_id,
        fixed_warmup_bars,
        lineage,
        validity,
    }
}

use ResidentQuantRouteLineageV3::{
    V2BitwisePreserved, V3AnnualizedExactLogMigration, V3ExactLogMigration,
    V3TemporalSessionMigration,
};
use ResidentQuantValidityRuleV3::{
    AlwaysValid, AsianOpeningRangeObservedBars, CumulativeDeltaPrefixOrZeroDenominator,
    FixedWarmup, FixedWarmupOrZeroDenominator, PreviousFiveTradingDaysOrZeroDenominator,
    PreviousUtcDayBoundaryOrZeroDenominator,
};

pub const RESIDENT_QUANT_ROUTE_CENSUS_V3: [ResidentQuantRouteCensusV3; 63] = [
    route("quant_close", "close", 0, V2BitwisePreserved, AlwaysValid),
    route(
        "quant_return_1",
        "simple_return:lag=1",
        1,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_return_2",
        "simple_return:lag=2",
        2,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_return_3",
        "simple_return:lag=3",
        3,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_return_5",
        "simple_return:lag=5",
        5,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_return_8",
        "simple_return:lag=8",
        8,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_return_13",
        "simple_return:lag=13",
        13,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_return_21",
        "simple_return:lag=21",
        21,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_log_return",
        "log_return:lag=1",
        1,
        V3ExactLogMigration,
        FixedWarmup,
    ),
    route(
        "quant_log_volatility",
        "log_range",
        0,
        V3ExactLogMigration,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_realized_vol_5",
        "realized_vol:window=5",
        5,
        V3AnnualizedExactLogMigration,
        FixedWarmup,
    ),
    route(
        "quant_realized_vol_10",
        "realized_vol:window=10",
        10,
        V3AnnualizedExactLogMigration,
        FixedWarmup,
    ),
    route(
        "quant_realized_vol_20",
        "realized_vol:window=20",
        20,
        V3AnnualizedExactLogMigration,
        FixedWarmup,
    ),
    route(
        "quant_realized_vol_50",
        "realized_vol:window=50",
        50,
        V3AnnualizedExactLogMigration,
        FixedWarmup,
    ),
    route(
        "quant_gk_vol_10",
        "garman_klass_vol:window=10",
        10,
        V3AnnualizedExactLogMigration,
        FixedWarmup,
    ),
    route(
        "quant_gk_vol_20",
        "garman_klass_vol:window=20",
        20,
        V3AnnualizedExactLogMigration,
        FixedWarmup,
    ),
    route(
        "quant_parkinson_vol_10",
        "parkinson_vol:window=10",
        10,
        V3AnnualizedExactLogMigration,
        FixedWarmup,
    ),
    route(
        "quant_parkinson_vol_20",
        "parkinson_vol:window=20",
        20,
        V3AnnualizedExactLogMigration,
        FixedWarmup,
    ),
    route(
        "quant_vol_ratio",
        "volatility_ratio:short=5;long=20",
        20,
        V3ExactLogMigration,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_hurst_100",
        "hurst_rescaled_range:window=100",
        100,
        V3ExactLogMigration,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_autocorr_1",
        "autocorrelation:window=50;lag=1",
        51,
        V3ExactLogMigration,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_autocorr_5",
        "autocorrelation:window=50;lag=5",
        55,
        V3ExactLogMigration,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_autocorr_10",
        "autocorrelation:window=50;lag=10",
        60,
        V3ExactLogMigration,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_efficiency_ratio_10",
        "kaufman_efficiency_ratio:window=10",
        10,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_efficiency_ratio_20",
        "kaufman_efficiency_ratio:window=20",
        20,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_skewness_30",
        "return_skewness:window=30",
        30,
        V3ExactLogMigration,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_kurtosis_30",
        "return_excess_kurtosis:window=30",
        30,
        V3ExactLogMigration,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_kyle_lambda",
        "kyle_lambda:window=20",
        20,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_vpin",
        "vpin:bucket=50;window=10",
        500,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_amihud_illiquidity",
        "amihud_illiquidity:window=20",
        20,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_roll_spread",
        "roll_spread:window=20",
        21,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_consec_up",
        "consecutive_up_bars",
        1,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_consec_down",
        "consecutive_down_bars",
        1,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_inside_bar",
        "inside_bar",
        1,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_outside_bar",
        "outside_bar",
        1,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_body_ratio",
        "body_to_range",
        0,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_upper_shadow",
        "upper_shadow_to_range",
        0,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_lower_shadow",
        "lower_shadow_to_range",
        0,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_prev_day_h_dist",
        "previous_utc_day_high_distance",
        0,
        V3TemporalSessionMigration,
        PreviousUtcDayBoundaryOrZeroDenominator,
    ),
    route(
        "quant_prev_day_l_dist",
        "previous_utc_day_low_distance",
        0,
        V3TemporalSessionMigration,
        PreviousUtcDayBoundaryOrZeroDenominator,
    ),
    route(
        "quant_prev_week_h_dist",
        "previous_five_trading_day_high_distance",
        0,
        V3TemporalSessionMigration,
        PreviousFiveTradingDaysOrZeroDenominator,
    ),
    route(
        "quant_prev_week_l_dist",
        "previous_five_trading_day_low_distance",
        0,
        V3TemporalSessionMigration,
        PreviousFiveTradingDaysOrZeroDenominator,
    ),
    route(
        "quant_orb_4",
        "asian_opening_range_breakout:bars=4",
        4,
        V3TemporalSessionMigration,
        AsianOpeningRangeObservedBars,
    ),
    route(
        "quant_orb_8",
        "asian_opening_range_breakout:bars=8",
        8,
        V3TemporalSessionMigration,
        AsianOpeningRangeObservedBars,
    ),
    route(
        "quant_orb_12",
        "asian_opening_range_breakout:bars=12",
        12,
        V3TemporalSessionMigration,
        AsianOpeningRangeObservedBars,
    ),
    route(
        "quant_amd_phase",
        "accumulation_manipulation_distribution:window=20",
        20,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_wyckoff",
        "wyckoff_phase:window=30",
        30,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_engulfing_vol",
        "engulfing_with_volume",
        1,
        V2BitwisePreserved,
        FixedWarmup,
    ),
    route(
        "quant_pivot_dist",
        "previous_utc_day_pivot_distance",
        0,
        V3TemporalSessionMigration,
        PreviousUtcDayBoundaryOrZeroDenominator,
    ),
    route(
        "quant_r1_dist",
        "previous_utc_day_r1_distance",
        0,
        V3TemporalSessionMigration,
        PreviousUtcDayBoundaryOrZeroDenominator,
    ),
    route(
        "quant_r2_dist",
        "previous_utc_day_r2_distance",
        0,
        V3TemporalSessionMigration,
        PreviousUtcDayBoundaryOrZeroDenominator,
    ),
    route(
        "quant_s1_dist",
        "previous_utc_day_s1_distance",
        0,
        V3TemporalSessionMigration,
        PreviousUtcDayBoundaryOrZeroDenominator,
    ),
    route(
        "quant_s2_dist",
        "previous_utc_day_s2_distance",
        0,
        V3TemporalSessionMigration,
        PreviousUtcDayBoundaryOrZeroDenominator,
    ),
    route(
        "quant_cam_r3_dist",
        "previous_utc_day_camarilla_r3_distance",
        0,
        V3TemporalSessionMigration,
        PreviousUtcDayBoundaryOrZeroDenominator,
    ),
    route(
        "quant_cam_s3_dist",
        "previous_utc_day_camarilla_s3_distance",
        0,
        V3TemporalSessionMigration,
        PreviousUtcDayBoundaryOrZeroDenominator,
    ),
    route(
        "quant_zscore_20",
        "close_zscore:window=20",
        20,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_zscore_50",
        "close_zscore:window=50",
        50,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_fractal_dim",
        "fractal_dimension:window=30",
        30,
        V3ExactLogMigration,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_rvol_10",
        "relative_volume:window=10",
        10,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_rvol_20",
        "relative_volume:window=20",
        20,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_rvol_50",
        "relative_volume:window=50",
        50,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_delta_volume",
        "delta_volume",
        0,
        V2BitwisePreserved,
        FixedWarmupOrZeroDenominator,
    ),
    route(
        "quant_cum_delta_zscore",
        "cumulative_delta_zscore:window=50",
        50,
        V2BitwisePreserved,
        CumulativeDeltaPrefixOrZeroDenominator,
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn census_is_complete_unique_and_in_schema_order() {
        assert_eq!(RESIDENT_QUANT_ROUTE_CENSUS_V3.len(), 63);
        assert_eq!(
            RESIDENT_QUANT_ROUTE_CENSUS_V3.map(|route| route.name),
            RESIDENT_QUANT_COLUMN_NAMES_V3
        );
        assert_eq!(
            RESIDENT_QUANT_COLUMN_NAMES_V3
                .into_iter()
                .collect::<BTreeSet<_>>()
                .len(),
            63
        );
    }

    #[test]
    fn lineage_is_exactly_31_plus_10_plus_8_plus_14() {
        let count = |lineage| {
            RESIDENT_QUANT_ROUTE_CENSUS_V3
                .iter()
                .filter(|route| route.lineage == lineage)
                .count()
        };
        assert_eq!(
            count(V2BitwisePreserved),
            RESIDENT_QUANT_V2_BITWISE_PRESERVED_ROUTE_COUNT_V3
        );
        assert_eq!(
            count(V3ExactLogMigration),
            RESIDENT_QUANT_V3_EXACT_LOG_MIGRATION_ROUTE_COUNT_V3
        );
        assert_eq!(
            count(V3AnnualizedExactLogMigration),
            RESIDENT_QUANT_V3_ANNUALIZED_EXACT_LOG_MIGRATION_ROUTE_COUNT_V3
        );
        assert_eq!(
            count(V3TemporalSessionMigration),
            RESIDENT_QUANT_V3_TEMPORAL_MIGRATION_ROUTE_COUNT_V3
        );
        assert_eq!(
            RESIDENT_QUANT_V2_BITWISE_PRESERVED_ROUTE_COUNT_V3
                + RESIDENT_QUANT_V3_MIGRATED_ROUTE_COUNT_V3,
            RESIDENT_QUANT_FEATURE_COLUMN_COUNT_V3
        );
        let changed = RESIDENT_QUANT_ROUTE_CENSUS_V3
            .iter()
            .filter(|route| route.lineage != V2BitwisePreserved)
            .map(|route| route.name)
            .collect::<Vec<_>>();
        assert_eq!(changed, RESIDENT_QUANT_V3_CHANGED_COLUMN_NAMES_V3);
    }
}
