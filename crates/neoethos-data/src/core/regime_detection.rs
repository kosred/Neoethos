//! Reviewed Regime semantic-v3 CPU authority.
//!
//! Semantic v2 is intentionally not callable from this module. Its anonymous
//! names and values are migration-refusal inputs only; regeneration from
//! canonical OHLC is the sole upgrade path.

use super::super::Ohlcv;
use super::regime_exact_math_v1::{
    REGIME_LOG49_OPERATION_TOKENS_SHA256_V1, REGIME_LOG49_OPERATION_TOKENS_V1,
    REGIME_LOG49_RUST_MIRROR_SHA256_V1, neoethos_ln_positive_exact_v1,
    neoethos_log10_positive_exact_v1,
};
use crate::core::features::{FeatureCellValidity, FeatureColumnF64};
use anyhow::Result;
use std::error::Error as StdError;
use std::fmt;

pub const REGIME_SEMANTIC_VERSION: u32 = 3;
pub const REGIME_OPERATION_SCHEDULE_V1: &str =
    "neoethos.regime.semantic-v3.f64-rn-fixed-order-log49-neumaier-v1";
pub const REGIME_SEMANTIC_V3_FIXTURE_SHA256: &str =
    "f0f89c26727e90206bb85bdb4b3f6e11f59652176f7ba8475e9fbaa301548a93";
pub const REGIME_V2_ARTIFACT_MIGRATION_POLICY: &str =
    "refuse semantic-v2 Regime artifacts and regenerate from canonical OHLC under semantic-v3";
pub const REGIME_CANONICAL_NAN_BITS_V3: u64 = 0x7ff8_0000_0000_0000;
pub const REGIME_COLUMN_COUNT_V3: usize = 14;
pub const REGIME_POINTER_TABLE_BYTES_V3: usize = 448;
pub const REGIME_ISOLATED_POINTER_SCHEMA_METADATA_BYTES_V3: usize = 1_235;

pub const REGIME_FEATURE_NAMES_V3: [&str; REGIME_COLUMN_COUNT_V3] = [
    "neoethos_custom_gk_vol_ratio_state_10_50_v3",
    "neoethos_custom_gk_vol_ratio_offset_10_50_v3",
    "regime_wilder_adx_14_v3",
    "neoethos_custom_wilder_di_dominance_direction_14_v3",
    "neoethos_custom_wilder_adx_direction_state_14_25_v3",
    "neoethos_custom_bollinger_keltner_squeeze_state_20_2_1p5_v3",
    "neoethos_custom_bollinger_midline_atr_deviation_20_v3",
    "neoethos_custom_directional_persistence_balance_20_v3",
    "neoethos_custom_candle_body_range_balance_8_v3",
    "regime_dreiss_choppiness_index_14_v3",
    "neoethos_custom_standardized_cusum_up_50_0p5_3_v3",
    "neoethos_custom_standardized_cusum_down_50_0p5_3_v3",
    "neoethos_custom_standardized_cusum_signal_50_0p5_3_v3",
    "neoethos_custom_equal_width_log_return_entropy_30_10_v3",
];

pub const REGIME_RETIRED_V2_FEATURE_NAMES: [&str; REGIME_COLUMN_COUNT_V3] = [
    "regime_vol_state",
    "regime_vol_zscore",
    "regime_trend_strength",
    "regime_trend_direction",
    "regime_trend_state",
    "regime_squeeze",
    "regime_squeeze_momentum",
    "regime_mr_vs_momentum",
    "regime_rei",
    "regime_choppiness",
    "regime_cusum_up",
    "regime_cusum_down",
    "regime_change_signal",
    "regime_entropy",
];

const LN_10_BITS_V3: u64 = 0x4002_6bb1_bbb5_5515;
const GK_COEFFICIENT_BITS_V3: u64 = 0x3fd8_b90b_fbe8_e7bc;
const ENTROPY_BIN_MULTIPLIER_BITS_V3: u64 = 0x4023_ff7c_ed91_6873;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegimeOhlcFieldV3 {
    Open,
    High,
    Low,
    Close,
}

impl fmt::Display for RegimeOhlcFieldV3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Open => "open",
            Self::High => "high",
            Self::Low => "low",
            Self::Close => "close",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegimeInputRefusalV3 {
    EmptyInput,
    LengthMismatch {
        close: usize,
        open: usize,
        high: usize,
        low: usize,
    },
    NonFiniteOhlc {
        row: usize,
        field: RegimeOhlcFieldV3,
    },
    NonPositiveOhlc {
        row: usize,
        field: RegimeOhlcFieldV3,
    },
    OhlcEnvelopeViolation {
        row: usize,
    },
    ScaleRangeUnsupported {
        row: usize,
        field: RegimeOhlcFieldV3,
    },
}

impl fmt::Display for RegimeInputRefusalV3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => formatter.write_str("RegimeInputRefusalV3::EmptyInput"),
            Self::LengthMismatch {
                close,
                open,
                high,
                low,
            } => write!(
                formatter,
                "RegimeInputRefusalV3::LengthMismatch{{close:{close},open:{open},high:{high},low:{low}}}"
            ),
            Self::NonFiniteOhlc { row, field } => write!(
                formatter,
                "RegimeInputRefusalV3::NonFiniteOhlc{{row:{row},field:{field}}}"
            ),
            Self::NonPositiveOhlc { row, field } => write!(
                formatter,
                "RegimeInputRefusalV3::NonPositiveOhlc{{row:{row},field:{field}}}"
            ),
            Self::OhlcEnvelopeViolation { row } => write!(
                formatter,
                "RegimeInputRefusalV3::OhlcEnvelopeViolation{{row:{row}}}"
            ),
            Self::ScaleRangeUnsupported { row, field } => write!(
                formatter,
                "RegimeInputRefusalV3::ScaleRangeUnsupported{{row:{row},field:{field}}}"
            ),
        }
    }
}

impl StdError for RegimeInputRefusalV3 {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RegimeInputAdmissionV3 {
    row_count: usize,
    scale_anchor: f64,
}

impl RegimeInputAdmissionV3 {
    pub(crate) const fn row_count(self) -> usize {
        self.row_count
    }

    pub(crate) const fn scale_anchor(self) -> f64 {
        self.scale_anchor
    }
}

fn binary_floor_exponent_v3(value: f64) -> i32 {
    let bits = value.to_bits();
    let stored = ((bits >> 52) & 0x7ff) as i32;
    if stored != 0 {
        return stored - 1023;
    }
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    let highest_bit = 63_i32 - fraction.leading_zeros() as i32;
    -1074 + highest_bit
}

fn exact_power_of_two_v3(exponent: i32) -> Option<f64> {
    if (-1022..=1023).contains(&exponent) {
        let stored = u64::try_from(exponent + 1023).ok()?;
        Some(f64::from_bits(stored << 52))
    } else if (-1074..=-1023).contains(&exponent) {
        let bit = u32::try_from(exponent + 1074).ok()?;
        Some(f64::from_bits(1_u64 << bit))
    } else {
        None
    }
}

/// Validate the complete producer input before allocating any output column.
pub(crate) fn admit_regime_input_v3(
    ohlcv: &Ohlcv,
) -> std::result::Result<RegimeInputAdmissionV3, RegimeInputRefusalV3> {
    let row_count = ohlcv.close.len();
    if row_count == 0 {
        return Err(RegimeInputRefusalV3::EmptyInput);
    }
    if ohlcv.open.len() != row_count
        || ohlcv.high.len() != row_count
        || ohlcv.low.len() != row_count
    {
        return Err(RegimeInputRefusalV3::LengthMismatch {
            close: row_count,
            open: ohlcv.open.len(),
            high: ohlcv.high.len(),
            low: ohlcv.low.len(),
        });
    }

    let mut greatest = 0.0_f64;
    let mut greatest_location = (0_usize, RegimeOhlcFieldV3::Open);
    for row in 0..row_count {
        let fields = [
            (RegimeOhlcFieldV3::Open, ohlcv.open[row]),
            (RegimeOhlcFieldV3::High, ohlcv.high[row]),
            (RegimeOhlcFieldV3::Low, ohlcv.low[row]),
            (RegimeOhlcFieldV3::Close, ohlcv.close[row]),
        ];
        for (field, value) in fields {
            if !value.is_finite() {
                return Err(RegimeInputRefusalV3::NonFiniteOhlc { row, field });
            }
            if value <= 0.0 {
                return Err(RegimeInputRefusalV3::NonPositiveOhlc { row, field });
            }
            if value > greatest {
                greatest = value;
                greatest_location = (row, field);
            }
        }
        if ohlcv.low[row] > ohlcv.open[row].min(ohlcv.close[row])
            || ohlcv.high[row] < ohlcv.open[row].max(ohlcv.close[row])
        {
            return Err(RegimeInputRefusalV3::OhlcEnvelopeViolation { row });
        }
    }

    let anchor_exponent = -binary_floor_exponent_v3(greatest);
    let scale_anchor = exact_power_of_two_v3(anchor_exponent).ok_or(
        RegimeInputRefusalV3::ScaleRangeUnsupported {
            row: greatest_location.0,
            field: greatest_location.1,
        },
    )?;
    for row in 0..row_count {
        for (field, value) in [
            (RegimeOhlcFieldV3::Open, ohlcv.open[row]),
            (RegimeOhlcFieldV3::High, ohlcv.high[row]),
            (RegimeOhlcFieldV3::Low, ohlcv.low[row]),
            (RegimeOhlcFieldV3::Close, ohlcv.close[row]),
        ] {
            let scaled = value * scale_anchor;
            if !scaled.is_finite() || scaled == 0.0 {
                return Err(RegimeInputRefusalV3::ScaleRangeUnsupported { row, field });
            }
        }
    }
    Ok(RegimeInputAdmissionV3 {
        row_count,
        scale_anchor,
    })
}

#[inline]
fn scaled_price_v3(value: f64, admission: RegimeInputAdmissionV3) -> f64 {
    value * admission.scale_anchor()
}

#[inline]
fn canonical_nan_v3() -> f64 {
    f64::from_bits(REGIME_CANONICAL_NAN_BITS_V3)
}

pub(crate) fn ordered_neumaier_sum_v1(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let next = sum + value;
        if sum.abs() >= value.abs() {
            compensation = compensation + ((sum - next) + value);
        } else {
            compensation = compensation + ((value - next) + sum);
        }
        sum = next;
    }
    sum + compensation
}

#[inline]
fn true_range_v3(ohlcv: &Ohlcv, row: usize, admission: RegimeInputAdmissionV3) -> f64 {
    let high = scaled_price_v3(ohlcv.high[row], admission);
    let low = scaled_price_v3(ohlcv.low[row], admission);
    let previous_close = scaled_price_v3(ohlcv.close[row - 1], admission);
    (high - low)
        .max((high - previous_close).abs())
        .max((low - previous_close).abs())
}

fn mark_invalid_v3(
    values: &mut [Vec<f64>; REGIME_COLUMN_COUNT_V3],
    validity: &mut [Vec<FeatureCellValidity>; REGIME_COLUMN_COUNT_V3],
    slot: usize,
    row: usize,
    reason: FeatureCellValidity,
) {
    debug_assert!(!reason.is_valid());
    values[slot][row] = canonical_nan_v3();
    validity[slot][row] = reason;
}

fn mark_valid_v3(
    values: &mut [Vec<f64>; REGIME_COLUMN_COUNT_V3],
    validity: &mut [Vec<FeatureCellValidity>; REGIME_COLUMN_COUNT_V3],
    slot: usize,
    row: usize,
    value: f64,
) {
    if value.is_finite() {
        values[slot][row] = value;
        validity[slot][row] = FeatureCellValidity::Valid;
    } else {
        mark_invalid_v3(
            values,
            validity,
            slot,
            row,
            FeatureCellValidity::ComputeFailure,
        );
    }
}

fn garman_klass_component_v3(
    ohlcv: &Ohlcv,
    row: usize,
    admission: RegimeInputAdmissionV3,
) -> Option<f64> {
    let open = scaled_price_v3(ohlcv.open[row], admission);
    let high = scaled_price_v3(ohlcv.high[row], admission);
    let low = scaled_price_v3(ohlcv.low[row], admission);
    let close = scaled_price_v3(ohlcv.close[row], admission);
    let u = neoethos_ln_positive_exact_v1(high) - neoethos_ln_positive_exact_v1(open);
    let d = neoethos_ln_positive_exact_v1(low) - neoethos_ln_positive_exact_v1(open);
    let c = neoethos_ln_positive_exact_v1(close) - neoethos_ln_positive_exact_v1(open);
    let range = u - d;
    let component = 0.5 * (range * range) - f64::from_bits(GK_COEFFICIENT_BITS_V3) * (c * c);
    (component.is_finite() && component >= 0.0).then_some(component)
}

fn compute_garman_klass_v3(
    ohlcv: &Ohlcv,
    admission: RegimeInputAdmissionV3,
    values: &mut [Vec<f64>; REGIME_COLUMN_COUNT_V3],
    validity: &mut [Vec<FeatureCellValidity>; REGIME_COLUMN_COUNT_V3],
) {
    for row in 49..admission.row_count() {
        let mut components = [0.0_f64; 50];
        let mut failed = false;
        for (offset, source_row) in ((row - 49)..=row).enumerate() {
            match garman_klass_component_v3(ohlcv, source_row, admission) {
                Some(component) => components[offset] = component,
                None => {
                    failed = true;
                    break;
                }
            }
        }
        if failed {
            for slot in 0..=1 {
                mark_invalid_v3(
                    values,
                    validity,
                    slot,
                    row,
                    FeatureCellValidity::ComputeFailure,
                );
            }
            continue;
        }
        let short_variance = ordered_neumaier_sum_v1(components[40..50].iter().copied()) / 10.0;
        let long_variance = ordered_neumaier_sum_v1(components.iter().copied()) / 50.0;
        if !short_variance.is_finite()
            || !long_variance.is_finite()
            || short_variance < 0.0
            || long_variance < 0.0
        {
            for slot in 0..=1 {
                mark_invalid_v3(
                    values,
                    validity,
                    slot,
                    row,
                    FeatureCellValidity::ComputeFailure,
                );
            }
            continue;
        }
        let short_gk = short_variance.sqrt();
        let long_gk = long_variance.sqrt();
        if long_gk == 0.0 {
            for slot in 0..=1 {
                mark_invalid_v3(
                    values,
                    validity,
                    slot,
                    row,
                    FeatureCellValidity::ZeroDenominator,
                );
            }
            continue;
        }
        let ratio = short_gk / long_gk;
        if !ratio.is_finite() {
            for slot in 0..=1 {
                mark_invalid_v3(
                    values,
                    validity,
                    slot,
                    row,
                    FeatureCellValidity::ComputeFailure,
                );
            }
            continue;
        }
        let state = if ratio > 1.5 {
            1.0
        } else if ratio < 0.6 {
            -1.0
        } else {
            0.0
        };
        let offset = (ratio - 1.0).max(-3.0).min(3.0);
        mark_valid_v3(values, validity, 0, row, state);
        mark_valid_v3(values, validity, 1, row, offset);
    }
}

fn compute_wilder_v3(
    ohlcv: &Ohlcv,
    admission: RegimeInputAdmissionV3,
    values: &mut [Vec<f64>; REGIME_COLUMN_COUNT_V3],
    validity: &mut [Vec<FeatureCellValidity>; REGIME_COLUMN_COUNT_V3],
) {
    let n = admission.row_count();
    if n <= 14 {
        return;
    }
    let mut plus_seed = [0.0_f64; 14];
    let mut minus_seed = [0.0_f64; 14];
    let mut tr_seed = [0.0_f64; 14];
    for row in 1..=14 {
        let high = scaled_price_v3(ohlcv.high[row], admission);
        let previous_high = scaled_price_v3(ohlcv.high[row - 1], admission);
        let low = scaled_price_v3(ohlcv.low[row], admission);
        let previous_low = scaled_price_v3(ohlcv.low[row - 1], admission);
        let up_move = high - previous_high;
        let down_move = previous_low - low;
        plus_seed[row - 1] = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        minus_seed[row - 1] = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };
        tr_seed[row - 1] = true_range_v3(ohlcv, row, admission);
    }
    let mut plus_smooth = ordered_neumaier_sum_v1(plus_seed);
    let mut minus_smooth = ordered_neumaier_sum_v1(minus_seed);
    let mut tr_smooth = ordered_neumaier_sum_v1(tr_seed);
    let mut dx_seed = [0.0_f64; 14];
    let mut dx_seed_count = 0_usize;
    let mut adx = 0.0_f64;
    let mut adx_live = false;

    for row in 14..n {
        if row > 14 {
            let high = scaled_price_v3(ohlcv.high[row], admission);
            let previous_high = scaled_price_v3(ohlcv.high[row - 1], admission);
            let low = scaled_price_v3(ohlcv.low[row], admission);
            let previous_low = scaled_price_v3(ohlcv.low[row - 1], admission);
            let up_move = high - previous_high;
            let down_move = previous_low - low;
            let plus_dm = if up_move > down_move && up_move > 0.0 {
                up_move
            } else {
                0.0
            };
            let minus_dm = if down_move > up_move && down_move > 0.0 {
                down_move
            } else {
                0.0
            };
            plus_smooth = (plus_smooth - plus_smooth / 14.0) + plus_dm;
            minus_smooth = (minus_smooth - minus_smooth / 14.0) + minus_dm;
            tr_smooth = (tr_smooth - tr_smooth / 14.0) + true_range_v3(ohlcv, row, admission);
        }

        let (direction, dx) =
            if !plus_smooth.is_finite() || !minus_smooth.is_finite() || !tr_smooth.is_finite() {
                (Err(FeatureCellValidity::ComputeFailure), None)
            } else if tr_smooth == 0.0 {
                (Err(FeatureCellValidity::ZeroDenominator), None)
            } else {
                let plus_di = (plus_smooth / tr_smooth) * 100.0;
                let minus_di = (minus_smooth / tr_smooth) * 100.0;
                let di_sum = plus_di + minus_di;
                if !plus_di.is_finite() || !minus_di.is_finite() || !di_sum.is_finite() {
                    (Err(FeatureCellValidity::ComputeFailure), None)
                } else if di_sum == 0.0 {
                    (Err(FeatureCellValidity::ZeroDenominator), None)
                } else {
                    let direction = if plus_di > minus_di {
                        1.0
                    } else if minus_di > plus_di {
                        -1.0
                    } else {
                        0.0
                    };
                    let dx = ((plus_di - minus_di).abs() / di_sum) * 100.0;
                    if dx.is_finite() {
                        (Ok(direction), Some(dx))
                    } else {
                        (Err(FeatureCellValidity::ComputeFailure), None)
                    }
                }
            };

        match direction {
            Ok(value) => mark_valid_v3(values, validity, 3, row, value),
            Err(reason) => mark_invalid_v3(values, validity, 3, row, reason),
        }

        let Some(dx) = dx else {
            dx_seed_count = 0;
            adx_live = false;
            if row >= 27 {
                let reason = match direction {
                    Err(FeatureCellValidity::ComputeFailure) => FeatureCellValidity::ComputeFailure,
                    _ => FeatureCellValidity::ZeroDenominator,
                };
                mark_invalid_v3(values, validity, 2, row, reason);
                mark_invalid_v3(values, validity, 4, row, reason);
            }
            continue;
        };

        if adx_live {
            adx = ((adx * 13.0) + dx) / 14.0;
        } else {
            dx_seed[dx_seed_count] = dx;
            dx_seed_count += 1;
            if dx_seed_count == 14 {
                adx = ordered_neumaier_sum_v1(dx_seed) / 14.0;
                adx_live = true;
                dx_seed_count = 0;
            }
        }
        if row < 27 {
            continue;
        }
        if !adx_live {
            mark_invalid_v3(
                values,
                validity,
                2,
                row,
                FeatureCellValidity::ZeroDenominator,
            );
            mark_invalid_v3(
                values,
                validity,
                4,
                row,
                FeatureCellValidity::ZeroDenominator,
            );
        } else if !adx.is_finite() {
            adx_live = false;
            mark_invalid_v3(
                values,
                validity,
                2,
                row,
                FeatureCellValidity::ComputeFailure,
            );
            mark_invalid_v3(
                values,
                validity,
                4,
                row,
                FeatureCellValidity::ComputeFailure,
            );
        } else {
            mark_valid_v3(values, validity, 2, row, adx);
            let state = if adx > 25.0 {
                direction.expect("valid DX retains valid direction")
            } else {
                0.0
            };
            mark_valid_v3(values, validity, 4, row, state);
        }
    }
}

fn compute_bollinger_keltner_v3(
    ohlcv: &Ohlcv,
    admission: RegimeInputAdmissionV3,
    values: &mut [Vec<f64>; REGIME_COLUMN_COUNT_V3],
    validity: &mut [Vec<FeatureCellValidity>; REGIME_COLUMN_COUNT_V3],
) {
    for row in 20..admission.row_count() {
        let start = row - 19;
        let mean = ordered_neumaier_sum_v1(
            (start..=row).map(|j| scaled_price_v3(ohlcv.close[j], admission)),
        ) / 20.0;
        let variance = ordered_neumaier_sum_v1((start..=row).map(|j| {
            let deviation = scaled_price_v3(ohlcv.close[j], admission) - mean;
            deviation * deviation
        })) / 20.0;
        let tr_sum =
            ordered_neumaier_sum_v1((start..=row).map(|j| true_range_v3(ohlcv, j, admission)));
        let atr = tr_sum / 20.0;
        if !mean.is_finite() || !variance.is_finite() || variance < 0.0 || !atr.is_finite() {
            for slot in 5..=6 {
                mark_invalid_v3(
                    values,
                    validity,
                    slot,
                    row,
                    FeatureCellValidity::ComputeFailure,
                );
            }
        } else if atr == 0.0 {
            for slot in 5..=6 {
                mark_invalid_v3(
                    values,
                    validity,
                    slot,
                    row,
                    FeatureCellValidity::ZeroDenominator,
                );
            }
        } else {
            let standard_deviation = variance.sqrt();
            let bb_upper = mean + 2.0 * standard_deviation;
            let bb_lower = mean - 2.0 * standard_deviation;
            let kc_upper = mean + 1.5 * atr;
            let kc_lower = mean - 1.5 * atr;
            let state = if bb_upper < kc_upper && bb_lower > kc_lower {
                1.0
            } else {
                -1.0
            };
            let deviation = (scaled_price_v3(ohlcv.close[row], admission) - mean) / atr;
            mark_valid_v3(values, validity, 5, row, state);
            mark_valid_v3(values, validity, 6, row, deviation);
        }
    }
}

fn compute_other_bounded_v3(
    ohlcv: &Ohlcv,
    admission: RegimeInputAdmissionV3,
    values: &mut [Vec<f64>; REGIME_COLUMN_COUNT_V3],
    validity: &mut [Vec<FeatureCellValidity>; REGIME_COLUMN_COUNT_V3],
) {
    for row in 21..admission.row_count() {
        let mut same = 0_u32;
        let mut reversal = 0_u32;
        for j in (row - 19)..=row {
            let current = scaled_price_v3(ohlcv.close[j], admission)
                - scaled_price_v3(ohlcv.close[j - 1], admission);
            let previous = scaled_price_v3(ohlcv.close[j - 1], admission)
                - scaled_price_v3(ohlcv.close[j - 2], admission);
            if (current > 0.0 && previous > 0.0) || (current < 0.0 && previous < 0.0) {
                same += 1;
            } else if (current > 0.0 && previous < 0.0) || (current < 0.0 && previous > 0.0) {
                reversal += 1;
            }
        }
        let total = same + reversal;
        if total == 0 {
            mark_invalid_v3(
                values,
                validity,
                7,
                row,
                FeatureCellValidity::ZeroDenominator,
            );
        } else {
            mark_valid_v3(
                values,
                validity,
                7,
                row,
                (f64::from(same) - f64::from(reversal)) / f64::from(total),
            );
        }
    }

    for row in 7..admission.row_count() {
        let start = row - 7;
        let body_sum = ordered_neumaier_sum_v1((start..=row).map(|j| {
            scaled_price_v3(ohlcv.close[j], admission) - scaled_price_v3(ohlcv.open[j], admission)
        }));
        let range_sum = ordered_neumaier_sum_v1((start..=row).map(|j| {
            scaled_price_v3(ohlcv.high[j], admission) - scaled_price_v3(ohlcv.low[j], admission)
        }));
        if !body_sum.is_finite() || !range_sum.is_finite() {
            mark_invalid_v3(
                values,
                validity,
                8,
                row,
                FeatureCellValidity::ComputeFailure,
            );
        } else if range_sum == 0.0 {
            mark_invalid_v3(
                values,
                validity,
                8,
                row,
                FeatureCellValidity::ZeroDenominator,
            );
        } else {
            mark_valid_v3(
                values,
                validity,
                8,
                row,
                (body_sum / range_sum).max(-1.0).min(1.0),
            );
        }
    }

    let ln_10 = f64::from_bits(LN_10_BITS_V3);
    let log10_14 = neoethos_log10_positive_exact_v1(14.0, ln_10);
    for row in 14..admission.row_count() {
        let start = row - 13;
        let tr_sum =
            ordered_neumaier_sum_v1((start..=row).map(|j| true_range_v3(ohlcv, j, admission)));
        let mut highest_true_high = f64::NEG_INFINITY;
        let mut lowest_true_low = f64::INFINITY;
        for j in start..=row {
            let high = scaled_price_v3(ohlcv.high[j], admission);
            let low = scaled_price_v3(ohlcv.low[j], admission);
            let previous_close = scaled_price_v3(ohlcv.close[j - 1], admission);
            highest_true_high = highest_true_high.max(high.max(previous_close));
            lowest_true_low = lowest_true_low.min(low.min(previous_close));
        }
        let denominator = highest_true_high - lowest_true_low;
        if !tr_sum.is_finite() || !denominator.is_finite() {
            mark_invalid_v3(
                values,
                validity,
                9,
                row,
                FeatureCellValidity::ComputeFailure,
            );
        } else if tr_sum == 0.0 || denominator == 0.0 {
            mark_invalid_v3(
                values,
                validity,
                9,
                row,
                FeatureCellValidity::ZeroDenominator,
            );
        } else {
            let ratio = tr_sum / denominator;
            if ratio <= 0.0 || !ratio.is_finite() {
                mark_invalid_v3(
                    values,
                    validity,
                    9,
                    row,
                    FeatureCellValidity::ComputeFailure,
                );
            } else {
                let chop = (100.0 * neoethos_log10_positive_exact_v1(ratio, ln_10)) / log10_14;
                mark_valid_v3(values, validity, 9, row, chop);
            }
        }
    }
}

fn compute_cusum_v3(
    ohlcv: &Ohlcv,
    admission: RegimeInputAdmissionV3,
    values: &mut [Vec<f64>; REGIME_COLUMN_COUNT_V3],
    validity: &mut [Vec<FeatureCellValidity>; REGIME_COLUMN_COUNT_V3],
) {
    let mut previous_up = 0.0_f64;
    let mut previous_down = 0.0_f64;
    for row in 50..admission.row_count() {
        let mean = ordered_neumaier_sum_v1(
            ((row - 50)..row).map(|j| scaled_price_v3(ohlcv.close[j], admission)),
        ) / 50.0;
        let variance = ordered_neumaier_sum_v1(((row - 50)..row).map(|j| {
            let deviation = scaled_price_v3(ohlcv.close[j], admission) - mean;
            deviation * deviation
        })) / 49.0;
        let reason = if !mean.is_finite() || !variance.is_finite() || variance < 0.0 {
            Some(FeatureCellValidity::ComputeFailure)
        } else if variance == 0.0 {
            Some(FeatureCellValidity::ZeroDenominator)
        } else {
            None
        };
        if let Some(reason) = reason {
            previous_up = 0.0;
            previous_down = 0.0;
            for slot in 10..=12 {
                mark_invalid_v3(values, validity, slot, row, reason);
            }
            continue;
        }
        let standard_deviation = variance.sqrt();
        let z = (scaled_price_v3(ohlcv.close[row], admission) - mean) / standard_deviation;
        let raw_up = (previous_up + z) - 0.5;
        let raw_down = (previous_down - z) - 0.5;
        let candidate_up = if raw_up > 0.0 { raw_up } else { 0.0 };
        let candidate_down = if raw_down > 0.0 { raw_down } else { 0.0 };
        if !standard_deviation.is_finite()
            || !z.is_finite()
            || !candidate_up.is_finite()
            || !candidate_down.is_finite()
        {
            previous_up = 0.0;
            previous_down = 0.0;
            for slot in 10..=12 {
                mark_invalid_v3(
                    values,
                    validity,
                    slot,
                    row,
                    FeatureCellValidity::ComputeFailure,
                );
            }
            continue;
        }
        let (up, down, signal) = if candidate_up > 3.0 {
            (0.0, candidate_down, 1.0)
        } else if candidate_down > 3.0 {
            (candidate_up, 0.0, -1.0)
        } else {
            (candidate_up, candidate_down, 0.0)
        };
        previous_up = up;
        previous_down = down;
        mark_valid_v3(values, validity, 10, row, up);
        mark_valid_v3(values, validity, 11, row, down);
        mark_valid_v3(values, validity, 12, row, signal);
    }
}

fn compute_entropy_v3(
    ohlcv: &Ohlcv,
    admission: RegimeInputAdmissionV3,
    values: &mut [Vec<f64>; REGIME_COLUMN_COUNT_V3],
    validity: &mut [Vec<FeatureCellValidity>; REGIME_COLUMN_COUNT_V3],
) {
    let ln_10 = f64::from_bits(LN_10_BITS_V3);
    let bin_multiplier = f64::from_bits(ENTROPY_BIN_MULTIPLIER_BITS_V3);
    for row in 30..admission.row_count() {
        let mut returns = [0.0_f64; 30];
        let mut failed = false;
        for (offset, j) in ((row - 29)..=row).enumerate() {
            let current = scaled_price_v3(ohlcv.close[j], admission);
            let previous = scaled_price_v3(ohlcv.close[j - 1], admission);
            let value =
                neoethos_ln_positive_exact_v1(current) - neoethos_ln_positive_exact_v1(previous);
            if !value.is_finite() {
                failed = true;
                break;
            }
            returns[offset] = value;
        }
        if failed {
            mark_invalid_v3(
                values,
                validity,
                13,
                row,
                FeatureCellValidity::ComputeFailure,
            );
            continue;
        }
        let mut minimum = returns[0];
        let mut maximum = returns[0];
        for value in returns.iter().copied().skip(1) {
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        let range = maximum - minimum;
        if !range.is_finite() || range < 0.0 {
            mark_invalid_v3(
                values,
                validity,
                13,
                row,
                FeatureCellValidity::ComputeFailure,
            );
        } else if range == 0.0 {
            mark_valid_v3(values, validity, 13, row, 0.0);
        } else {
            let mut bins = [0_u32; 10];
            for value in returns {
                let coordinate = ((value - minimum) / range) * bin_multiplier;
                if !coordinate.is_finite() || coordinate < 0.0 {
                    failed = true;
                    break;
                }
                let bin = (coordinate as usize).min(9);
                bins[bin] += 1;
            }
            if failed {
                mark_invalid_v3(
                    values,
                    validity,
                    13,
                    row,
                    FeatureCellValidity::ComputeFailure,
                );
                continue;
            }
            let entropy_sum = ordered_neumaier_sum_v1(bins.into_iter().map(|count| {
                if count == 0 {
                    0.0
                } else {
                    let probability = f64::from(count) / 30.0;
                    probability * neoethos_ln_positive_exact_v1(probability)
                }
            }));
            mark_valid_v3(values, validity, 13, row, -entropy_sum / ln_10);
        }
    }
}

/// Exact f64 Regime-v3 CPU oracle with explicit logical validity.
pub fn compute_regime_feature_columns_f64(ohlcv: &Ohlcv) -> Result<Vec<FeatureColumnF64>> {
    let admission = admit_regime_input_v3(ohlcv)?;
    debug_assert!(!REGIME_LOG49_OPERATION_TOKENS_V1.is_empty());
    debug_assert_eq!(REGIME_LOG49_OPERATION_TOKENS_SHA256_V1.len(), 64);
    debug_assert_eq!(REGIME_LOG49_RUST_MIRROR_SHA256_V1.len(), 64);

    let mut values: [Vec<f64>; REGIME_COLUMN_COUNT_V3] =
        std::array::from_fn(|_| vec![canonical_nan_v3(); admission.row_count()]);
    let mut validity: [Vec<FeatureCellValidity>; REGIME_COLUMN_COUNT_V3] =
        std::array::from_fn(|_| vec![FeatureCellValidity::Warmup; admission.row_count()]);

    compute_garman_klass_v3(ohlcv, admission, &mut values, &mut validity);
    compute_wilder_v3(ohlcv, admission, &mut values, &mut validity);
    compute_bollinger_keltner_v3(ohlcv, admission, &mut values, &mut validity);
    compute_other_bounded_v3(ohlcv, admission, &mut values, &mut validity);
    compute_cusum_v3(ohlcv, admission, &mut values, &mut validity);
    compute_entropy_v3(ohlcv, admission, &mut values, &mut validity);

    REGIME_FEATURE_NAMES_V3
        .into_iter()
        .zip(values)
        .zip(validity)
        .map(|((name, values), validity)| FeatureColumnF64::new(name, values, validity))
        .collect()
}
