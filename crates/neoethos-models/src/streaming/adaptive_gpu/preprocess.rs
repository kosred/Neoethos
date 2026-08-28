use cubecl::prelude::*;

use super::{
    CLASS_COUNT, DEVICE_LABEL_MAP_FAULT, DEVICE_MISSING_CLASS_0_FAULT,
    DEVICE_MISSING_CLASS_1_FAULT, DEVICE_MISSING_CLASS_2_FAULT, DEVICE_SCALER_ARITHMETIC_FAULT,
    DEVICE_SCALER_INPUT_FAULT, DEVICE_SCALER_OUTPUT_FAULT, DEVICE_TRANSFORM_ARITHMETIC_FAULT,
};

pub(super) const LABEL_CHANNEL_COUNT: usize = CLASS_COUNT + 1;
pub(super) const LABEL_ROWS_PER_PARTIAL: usize = 1024;
pub(super) const SCALER_ROWS_PER_PARTIAL: usize = 1024;
pub(super) const TRANSFORM_ELEMENTS_PER_WORK_ITEM: usize = 64;
pub(super) const TRANSFORM_FAULTS_PER_PARTIAL: usize = 1024;

#[cube(launch)]
pub(super) fn online_pa_full_pipeline_initialize_v2_kernel(
    weights: &mut Array<f64>,
    bias: &mut Array<f64>,
    arithmetic_status: &mut Array<u32>,
    weight_count: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos < weight_count as usize {
        weights[pos] = 0.0;
    }
    if pos < CLASS_COUNT {
        bias[pos] = 0.0;
    }
    if pos == 0 {
        arithmetic_status[0] = 0;
    }
}

/// Map one original label per CUDA work item and emit four deterministic
/// indicators: class 0, class 1, class 2, invalid. No atomics are used.
#[cube(launch)]
pub(super) fn online_pa_original_label_map_v2_kernel(
    original_labels: &Array<i32>,
    remapped_labels: &mut Array<i32>,
    label_indicators: &mut Array<u32>,
    rows: u32,
) {
    if ABSOLUTE_POS < rows as usize {
        let row = ABSOLUTE_POS;
        let indicator_base = row * LABEL_CHANNEL_COUNT;
        label_indicators[indicator_base] = 0;
        label_indicators[indicator_base + 1] = 0;
        label_indicators[indicator_base + 2] = 0;
        label_indicators[indicator_base + 3] = 0;
        let original = original_labels[row];
        if original == 0 {
            remapped_labels[row] = 0;
            label_indicators[indicator_base] = 1;
        } else if original == 1 {
            remapped_labels[row] = 1;
            label_indicators[indicator_base + 1] = 1;
        } else if original == -1 {
            remapped_labels[row] = 2;
            label_indicators[indicator_base + 2] = 1;
        } else {
            remapped_labels[row] = 0;
            label_indicators[indicator_base + 3] = 1;
        }
    }
}

/// Each lane reduces at most LABEL_ROWS_PER_PARTIAL rows for one channel.
#[cube(launch)]
pub(super) fn online_pa_label_count_partial_v2_kernel(
    label_indicators: &Array<u32>,
    label_count_partials: &mut Array<u32>,
    rows: u32,
    partial_channel_count: u32,
) {
    if ABSOLUTE_POS < partial_channel_count as usize {
        let partial_channel = ABSOLUTE_POS;
        let channel = partial_channel % LABEL_CHANNEL_COUNT;
        let partial_index = partial_channel / LABEL_CHANNEL_COUNT;
        let start = partial_index * LABEL_ROWS_PER_PARTIAL;
        let total = RuntimeCell::<u32>::new(0);
        for offset in 0..LABEL_ROWS_PER_PARTIAL {
            let row = start + offset;
            if row < rows as usize {
                total.store(total.read() + label_indicators[row * LABEL_CHANNEL_COUNT + channel]);
            }
        }
        label_count_partials[partial_channel] = total.read();
    }
}

/// Four lanes finalize class counts, invalid-label status and clipped balanced
/// PA-I slack caps. Each lane combines partials in ascending chunk order.
#[cube(launch)]
pub(super) fn online_pa_label_count_weight_finalize_v2_kernel(
    label_count_partials: &Array<u32>,
    class_counts: &mut Array<u32>,
    class_weights: &mut Array<f64>,
    label_faults: &mut Array<u32>,
    partial_count: u32,
    rows: u32,
    finite_limit: f64,
) {
    if ABSOLUTE_POS < LABEL_CHANNEL_COUNT {
        let channel = ABSOLUTE_POS;
        let total = RuntimeCell::<u32>::new(0);
        for partial_index in 0..partial_count as usize {
            total.store(
                total.read() + label_count_partials[partial_index * LABEL_CHANNEL_COUNT + channel],
            );
        }
        let fault = RuntimeCell::<u32>::new(0);
        if channel == CLASS_COUNT {
            if total.read() != 0 {
                fault.store(DEVICE_LABEL_MAP_FAULT);
            }
        } else {
            class_counts[channel] = total.read();
            if total.read() == 0 {
                if channel == 0 {
                    fault.store(DEVICE_MISSING_CLASS_0_FAULT);
                } else if channel == 1 {
                    fault.store(DEVICE_MISSING_CLASS_1_FAULT);
                } else {
                    fault.store(DEVICE_MISSING_CLASS_2_FAULT);
                }
                class_weights[channel] = 0.0;
            } else {
                let weight = RuntimeCell::<f64>::new(
                    rows as f64 / (CLASS_COUNT as f64 * total.read() as f64),
                );
                if weight.read() < 0.5 {
                    weight.store(0.5);
                }
                if weight.read() > 4.0 {
                    weight.store(4.0);
                }
                if weight.read() <= finite_limit && weight.read() >= 0.5 {
                    class_weights[channel] = weight.read();
                } else {
                    class_weights[channel] = 0.0;
                    fault.store(DEVICE_LABEL_MAP_FAULT);
                }
            }
        }
        label_faults[channel] = fault.read();
    }
}

/// Each work item runs canonical Welford order over at most 1024 rows for one
/// column. The partial grid is parallel across both chunks and columns.
#[cube(launch)]
pub(super) fn online_pa_ddof0_scaler_partial_v2_kernel(
    raw_features: &Array<f64>,
    partial_means: &mut Array<f64>,
    partial_m2s: &mut Array<f64>,
    partial_faults: &mut Array<u32>,
    rows: u32,
    cols: u32,
    partial_value_count: u32,
    finite_limit: f64,
) {
    if ABSOLUTE_POS < partial_value_count as usize {
        let partial_position = ABSOLUTE_POS;
        let cols_us = cols as usize;
        let col = partial_position % cols_us;
        let partial_index = partial_position / cols_us;
        let start = partial_index * SCALER_ROWS_PER_PARTIAL;
        let count = RuntimeCell::<u32>::new(0);
        let mean = RuntimeCell::<f64>::new(0.0);
        let m2 = RuntimeCell::<f64>::new(0.0);
        let fault = RuntimeCell::<u32>::new(0);
        for offset in 0..SCALER_ROWS_PER_PARTIAL {
            let row = start + offset;
            if row < rows as usize && fault.read() == 0 {
                let value = raw_features[row * cols_us + col];
                if value <= finite_limit && value >= -finite_limit {
                    let next_count = count.read() + 1;
                    let delta = value - mean.read();
                    let next_mean = mean.read() + delta / next_count as f64;
                    let delta_after = value - next_mean;
                    let contribution = delta * delta_after;
                    let next_m2 = m2.read() + contribution;
                    let valid = delta <= finite_limit
                        && delta >= -finite_limit
                        && next_mean <= finite_limit
                        && next_mean >= -finite_limit
                        && delta_after <= finite_limit
                        && delta_after >= -finite_limit
                        && contribution <= finite_limit
                        && contribution >= -finite_limit
                        && next_m2 <= finite_limit
                        && next_m2 >= 0.0;
                    if valid {
                        count.store(next_count);
                        mean.store(next_mean);
                        m2.store(next_m2);
                    } else {
                        fault.store(DEVICE_SCALER_ARITHMETIC_FAULT);
                    }
                } else {
                    fault.store(DEVICE_SCALER_INPUT_FAULT);
                }
            }
        }
        partial_means[partial_position] = mean.read();
        partial_m2s[partial_position] = m2.read();
        partial_faults[partial_position] = fault.read();
    }
}

/// One lane per column combines bounded partials in ascending row-chunk order
/// with Chan's pairwise population-variance identity (ddof=0).
#[cube(launch)]
pub(super) fn online_pa_ddof0_scaler_finalize_v2_kernel(
    partial_means: &Array<f64>,
    partial_m2s: &Array<f64>,
    partial_faults: &Array<u32>,
    scaler_means: &mut Array<f64>,
    scaler_stds: &mut Array<f64>,
    scaler_faults: &mut Array<u32>,
    rows: u32,
    cols: u32,
    partial_count: u32,
    finite_limit: f64,
    minimum_std: f64,
) {
    if ABSOLUTE_POS < cols as usize {
        let col = ABSOLUTE_POS;
        let cols_us = cols as usize;
        let total_count = RuntimeCell::<u32>::new(0);
        let mean = RuntimeCell::<f64>::new(0.0);
        let m2 = RuntimeCell::<f64>::new(0.0);
        let fault = RuntimeCell::<u32>::new(0);
        for partial_index in 0..partial_count as usize {
            let position = partial_index * cols_us + col;
            if fault.read() == 0 {
                let partial_fault = partial_faults[position];
                if partial_fault != 0 {
                    fault.store(partial_fault);
                } else {
                    let start = partial_index * SCALER_ROWS_PER_PARTIAL;
                    let partial_count_value =
                        RuntimeCell::<u32>::new(SCALER_ROWS_PER_PARTIAL as u32);
                    if start + SCALER_ROWS_PER_PARTIAL > rows as usize {
                        partial_count_value.store((rows as usize - start) as u32);
                    }
                    let combined_count = total_count.read() + partial_count_value.read();
                    let partial_mean = partial_means[position];
                    let partial_m2 = partial_m2s[position];
                    let delta = partial_mean - mean.read();
                    let count_product =
                        total_count.read() as f64 * partial_count_value.read() as f64;
                    let next_mean = mean.read()
                        + delta * partial_count_value.read() as f64 / combined_count as f64;
                    let cross = delta * delta * count_product / combined_count as f64;
                    let next_m2 = m2.read() + partial_m2 + cross;
                    let valid = partial_mean <= finite_limit
                        && partial_mean >= -finite_limit
                        && partial_m2 <= finite_limit
                        && partial_m2 >= 0.0
                        && delta <= finite_limit
                        && delta >= -finite_limit
                        && count_product <= finite_limit
                        && count_product >= 0.0
                        && next_mean <= finite_limit
                        && next_mean >= -finite_limit
                        && cross <= finite_limit
                        && cross >= 0.0
                        && next_m2 <= finite_limit
                        && next_m2 >= 0.0;
                    if valid {
                        total_count.store(combined_count);
                        mean.store(next_mean);
                        m2.store(next_m2);
                    } else {
                        fault.store(DEVICE_SCALER_ARITHMETIC_FAULT);
                    }
                }
            }
        }
        if fault.read() == 0 && total_count.read() == rows {
            let variance = m2.read() / rows as f64;
            if variance <= finite_limit && variance >= 0.0 {
                let standard_deviation = variance.sqrt();
                if standard_deviation <= finite_limit && standard_deviation >= 0.0 {
                    scaler_means[col] = mean.read();
                    scaler_stds[col] = if standard_deviation > minimum_std {
                        standard_deviation
                    } else {
                        1.0.into()
                    };
                } else {
                    fault.store(DEVICE_SCALER_OUTPUT_FAULT);
                }
            } else {
                fault.store(DEVICE_SCALER_OUTPUT_FAULT);
            }
        } else if fault.read() == 0 {
            fault.store(DEVICE_SCALER_OUTPUT_FAULT);
        }
        if fault.read() != 0 {
            scaler_means[col] = 0.0;
            scaler_stds[col] = 1.0;
        }
        scaler_faults[col] = fault.read();
    }
}

#[cube(launch)]
pub(super) fn online_pa_scaler_transform_chunked_v2_kernel(
    raw_features: &Array<f64>,
    scaler_means: &Array<f64>,
    scaler_stds: &Array<f64>,
    scaled_features: &mut Array<f64>,
    transform_work_faults: &mut Array<u32>,
    feature_count: u32,
    work_item_count: u32,
    cols: u32,
    finite_limit: f64,
) {
    if ABSOLUTE_POS < work_item_count as usize {
        let work_item = ABSOLUTE_POS;
        let start = work_item * TRANSFORM_ELEMENTS_PER_WORK_ITEM;
        let fault = RuntimeCell::<u32>::new(0);
        for offset in 0..TRANSFORM_ELEMENTS_PER_WORK_ITEM {
            let pos = start + offset;
            if pos < feature_count as usize {
                let col = pos % cols as usize;
                let value = raw_features[pos];
                let mean = scaler_means[col];
                let standard_deviation = scaler_stds[col];
                let scaled = (value - mean) / standard_deviation;
                let valid = value <= finite_limit
                    && value >= -finite_limit
                    && mean <= finite_limit
                    && mean >= -finite_limit
                    && standard_deviation <= finite_limit
                    && standard_deviation > 0.0
                    && scaled <= finite_limit
                    && scaled >= -finite_limit;
                if valid {
                    scaled_features[pos] = scaled;
                } else {
                    scaled_features[pos] = 0.0;
                    fault.store(DEVICE_TRANSFORM_ARITHMETIC_FAULT);
                }
            }
        }
        transform_work_faults[work_item] = fault.read();
    }
}

#[cube(launch)]
pub(super) fn online_pa_transform_fault_partial_v2_kernel(
    transform_work_faults: &Array<u32>,
    transform_fault_partials: &mut Array<u32>,
    work_item_count: u32,
    fault_partial_count: u32,
) {
    if ABSOLUTE_POS < fault_partial_count as usize {
        let partial_index = ABSOLUTE_POS;
        let start = partial_index * TRANSFORM_FAULTS_PER_PARTIAL;
        let fault = RuntimeCell::<u32>::new(0);
        for offset in 0..TRANSFORM_FAULTS_PER_PARTIAL {
            let position = start + offset;
            if position < work_item_count as usize
                && fault.read() == 0
                && transform_work_faults[position] != 0
            {
                fault.store(transform_work_faults[position]);
            }
        }
        transform_fault_partials[partial_index] = fault.read();
    }
}

/// The only single-lane fault scan is bounded by four label channels, the
/// narrow column count, and one entry per 65,536 transformed elements.
#[cube(launch)]
pub(super) fn online_pa_preprocess_fault_finalize_v2_kernel(
    label_faults: &Array<u32>,
    scaler_faults: &Array<u32>,
    transform_fault_partials: &Array<u32>,
    arithmetic_status: &mut Array<u32>,
    cols: u32,
    transform_fault_partial_count: u32,
) {
    if ABSOLUTE_POS == 0 {
        let status = RuntimeCell::<u32>::new(label_faults[CLASS_COUNT]);
        for channel in 0..CLASS_COUNT {
            if status.read() == 0 && label_faults[channel] != 0 {
                status.store(label_faults[channel]);
            }
        }
        for col in 0..cols as usize {
            if status.read() == 0 && scaler_faults[col] != 0 {
                status.store(scaler_faults[col]);
            }
        }
        for partial in 0..transform_fault_partial_count as usize {
            if status.read() == 0 && transform_fault_partials[partial] != 0 {
                status.store(transform_fault_partials[partial]);
            }
        }
        arithmetic_status[0] = status.read();
    }
}
