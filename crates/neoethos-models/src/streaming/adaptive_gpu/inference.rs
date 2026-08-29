use cubecl::prelude::*;

use super::{CLASS_COUNT, DEVICE_INFERENCE_ARITHMETIC_FAULT, PA_CUBE_UNITS};

const SHARED_NORM_OFFSET: usize = 0;
const SHARED_SCORE_0_OFFSET: usize = PA_CUBE_UNITS;
const SHARED_SCORE_1_OFFSET: usize = PA_CUBE_UNITS * 2;
const SHARED_SCORE_2_OFFSET: usize = PA_CUBE_UNITS * 3;
const SHARED_FAULT_OFFSET: usize = PA_CUBE_UNITS * 4;
const SHARED_VALUES: usize = SHARED_FAULT_OFFSET + PA_CUBE_UNITS;

#[cube]
fn reduce_inference_partials(shared: &mut SharedMemory<f64>, unit: usize, stride: usize) {
    if unit < stride {
        shared[SHARED_SCORE_0_OFFSET + unit] =
            shared[SHARED_SCORE_0_OFFSET + unit] + shared[SHARED_SCORE_0_OFFSET + unit + stride];
        shared[SHARED_SCORE_1_OFFSET + unit] =
            shared[SHARED_SCORE_1_OFFSET + unit] + shared[SHARED_SCORE_1_OFFSET + unit + stride];
        shared[SHARED_SCORE_2_OFFSET + unit] =
            shared[SHARED_SCORE_2_OFFSET + unit] + shared[SHARED_SCORE_2_OFFSET + unit + stride];
        shared[SHARED_FAULT_OFFSET + unit] =
            shared[SHARED_FAULT_OFFSET + unit] + shared[SHARED_FAULT_OFFSET + unit + stride];
    }
    sync_cube();
}

/// Each cube owns one row. Its 256 lanes fuse raw scaling, three logits and
/// stable score-softmax; only probabilities and fault status leave the card.
#[cube(launch)]
pub(super) fn online_pa_fused_raw_scale_logits_softmax_v1_kernel(
    raw_features: &Array<f64>,
    scaler_means: &Array<f64>,
    scaler_stds: &Array<f64>,
    weights: &Array<f64>,
    bias: &Array<f64>,
    probabilities: &mut Array<f64>,
    row_status: &mut Array<u32>,
    rows: u32,
    cols: u32,
    finite_limit: f64,
) {
    let row = CUBE_POS;
    if row >= rows as usize {
        terminate!();
    }
    let unit = UNIT_POS as usize;
    let cols_us = cols as usize;
    let row_base = row * cols_us;
    let score_0 = RuntimeCell::<f64>::new(0.0);
    let score_1 = RuntimeCell::<f64>::new(0.0);
    let score_2 = RuntimeCell::<f64>::new(0.0);
    let local_fault = RuntimeCell::<f64>::new(0.0);
    let mut shared = SharedMemory::<f64>::new(SHARED_VALUES);

    for col in range_stepped(unit, cols_us, PA_CUBE_UNITS) {
        let value = raw_features[row_base + col];
        let mean = scaler_means[col];
        let standard_deviation = scaler_stds[col];
        let scaled = (value - mean) / standard_deviation;
        let next_0 = score_0.read() + scaled * weights[col];
        let next_1 = score_1.read() + scaled * weights[cols_us + col];
        let next_2 = score_2.read() + scaled * weights[cols_us * 2 + col];
        let valid = value <= finite_limit
            && value >= -finite_limit
            && mean <= finite_limit
            && mean >= -finite_limit
            && standard_deviation <= finite_limit
            && standard_deviation > 0.0
            && scaled <= finite_limit
            && scaled >= -finite_limit
            && next_0 <= finite_limit
            && next_0 >= -finite_limit
            && next_1 <= finite_limit
            && next_1 >= -finite_limit
            && next_2 <= finite_limit
            && next_2 >= -finite_limit;
        if valid {
            score_0.store(next_0);
            score_1.store(next_1);
            score_2.store(next_2);
        } else {
            local_fault.store(1.0);
        }
    }
    shared[SHARED_NORM_OFFSET + unit] = 0.0;
    shared[SHARED_SCORE_0_OFFSET + unit] = score_0.read();
    shared[SHARED_SCORE_1_OFFSET + unit] = score_1.read();
    shared[SHARED_SCORE_2_OFFSET + unit] = score_2.read();
    shared[SHARED_FAULT_OFFSET + unit] = local_fault.read();
    sync_cube();

    reduce_inference_partials(&mut shared, unit, 128);
    reduce_inference_partials(&mut shared, unit, 64);
    reduce_inference_partials(&mut shared, unit, 32);
    reduce_inference_partials(&mut shared, unit, 16);
    reduce_inference_partials(&mut shared, unit, 8);
    reduce_inference_partials(&mut shared, unit, 4);
    reduce_inference_partials(&mut shared, unit, 2);
    reduce_inference_partials(&mut shared, unit, 1);

    if unit == 0 {
        let logit_0 = shared[SHARED_SCORE_0_OFFSET] + bias[0];
        let logit_1 = shared[SHARED_SCORE_1_OFFSET] + bias[1];
        let logit_2 = shared[SHARED_SCORE_2_OFFSET] + bias[2];
        let fault = RuntimeCell::<u32>::new(0);
        let logits_finite = logit_0 <= finite_limit
            && logit_0 >= -finite_limit
            && logit_1 <= finite_limit
            && logit_1 >= -finite_limit
            && logit_2 <= finite_limit
            && logit_2 >= -finite_limit;
        if shared[SHARED_FAULT_OFFSET] > 0.0 || !logits_finite {
            fault.store(DEVICE_INFERENCE_ARITHMETIC_FAULT);
        }

        let maximum = RuntimeCell::<f64>::new(logit_0);
        if logit_1 > maximum.read() {
            maximum.store(logit_1);
        }
        if logit_2 > maximum.read() {
            maximum.store(logit_2);
        }
        let shifted_0 = logit_0 - maximum.read();
        let shifted_1 = logit_1 - maximum.read();
        let shifted_2 = logit_2 - maximum.read();
        let shifted_logits_finite = shifted_0 <= finite_limit
            && shifted_0 >= -finite_limit
            && shifted_1 <= finite_limit
            && shifted_1 >= -finite_limit
            && shifted_2 <= finite_limit
            && shifted_2 >= -finite_limit;
        let exp_0 = RuntimeCell::<f64>::new(0.0);
        let exp_1 = RuntimeCell::<f64>::new(0.0);
        let exp_2 = RuntimeCell::<f64>::new(0.0);
        if shifted_logits_finite && fault.read() == 0 {
            exp_0.store(shifted_0.exp());
            exp_1.store(shifted_1.exp());
            exp_2.store(shifted_2.exp());
        } else {
            fault.store(DEVICE_INFERENCE_ARITHMETIC_FAULT);
        }
        let normalizer = exp_0.read() + exp_1.read() + exp_2.read();
        let softmax_finite = exp_0.read() <= finite_limit
            && exp_0.read() >= 0.0
            && exp_1.read() <= finite_limit
            && exp_1.read() >= 0.0
            && exp_2.read() <= finite_limit
            && exp_2.read() >= 0.0
            && normalizer <= finite_limit
            && normalizer > 0.0;
        if !softmax_finite {
            fault.store(DEVICE_INFERENCE_ARITHMETIC_FAULT);
        }

        let output_base = row * CLASS_COUNT;
        if fault.read() == 0 {
            probabilities[output_base] = exp_0.read() / normalizer;
            probabilities[output_base + 1] = exp_1.read() / normalizer;
            probabilities[output_base + 2] = exp_2.read() / normalizer;
        } else {
            probabilities[output_base] = 0.0;
            probabilities[output_base + 1] = 0.0;
            probabilities[output_base + 2] = 0.0;
        }
        row_status[row] = fault.read();
    }
}
