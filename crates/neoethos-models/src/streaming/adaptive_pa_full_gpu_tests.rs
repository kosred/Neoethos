use anyhow::Result;
use ndarray::{Array1, Array2};
use std::time::{Duration, Instant};

use super::adaptive_gpu::{
    DEVICE_INFERENCE_ARITHMETIC_FAULT, try_fit_passive_aggressive_cuda_full_pipeline,
    try_predict_passive_aggressive_cuda_full_pipeline,
};
use super::adaptive_impl::clamped_balanced_class_slack_weights_v1;
use crate::statistical::common::{FeatureScaler, remap_three_class_labels, softmax_rows};

fn raw_fixture() -> (Array2<f64>, Vec<i32>) {
    let rows = 24;
    let features = Array2::from_shape_fn((rows, 3), |(row, col)| match col {
        0 => row as f64 / 11.0 - 1.0,
        1 => ((row * 5 % 13) as f64 - 6.0) / 4.0,
        _ => 7.0,
    });
    let labels = (0..rows)
        .map(|row| match row % 3 {
            0 => 0,
            1 => 1,
            _ => -1,
        })
        .collect();
    (features, labels)
}

fn prediction_based_weighted_slack_v2_cpu_oracle(
    features: &Array2<f64>,
    labels: &[usize],
    class_weights: &[f64; 3],
    aggressiveness: f64,
    epochs: usize,
) -> (Array2<f64>, Array1<f64>) {
    let mut weights = Array2::<f64>::zeros((3, features.ncols()));
    let mut bias = Array1::<f64>::zeros(3);
    for _ in 0..epochs {
        for (row, target) in labels.iter().copied().enumerate() {
            let x = features.row(row);
            let norm = x.iter().map(|value| value * value).sum::<f64>() + 1.0;
            let mut scores = [0.0_f64; 3];
            for class in 0..3 {
                scores[class] = weights
                    .row(class)
                    .iter()
                    .zip(x.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>()
                    + bias[class];
            }
            let mut predicted = 0;
            if scores[1] >= scores[predicted] {
                predicted = 1;
            }
            if scores[2] >= scores[predicted] {
                predicted = 2;
            }
            // JMLR section 8, equations 45 and 47-49: this explicit
            // prediction-based variant has zero loss on a correct argmax.
            if predicted == target {
                continue;
            }
            let margin = scores[predicted] - scores[target] + 1.0;
            if margin <= 0.0 {
                continue;
            }
            let tau = (margin / (2.0 * norm)).min(aggressiveness * class_weights[target]);
            for col in 0..features.ncols() {
                let delta = tau * x[col];
                weights[(target, col)] += delta;
                weights[(predicted, col)] -= delta;
            }
            bias[target] += tau;
            bias[predicted] -= tau;
        }
    }
    (weights, bias)
}

fn cpu_full_pipeline(
    raw: &Array2<f64>,
    original_labels: &[i32],
    inference_raw: &Array2<f64>,
    aggressiveness: f64,
    epochs: usize,
) -> Result<(
    FeatureScaler,
    [f64; 3],
    Array2<f64>,
    Array1<f64>,
    Array2<f64>,
)> {
    let labels = remap_three_class_labels(original_labels)?;
    let class_weights = clamped_balanced_class_slack_weights_v1(&labels)?;
    let scaler = FeatureScaler::fit(raw)?;
    let scaled = scaler.transform(raw)?;
    let (weights, bias) = prediction_based_weighted_slack_v2_cpu_oracle(
        &scaled,
        &labels,
        &class_weights,
        aggressiveness,
        epochs,
    );
    let inference_scaled = scaler.transform(inference_raw)?;
    let mut logits = Array2::<f64>::zeros((inference_scaled.nrows(), 3));
    for row in 0..inference_scaled.nrows() {
        for class in 0..3 {
            logits[(row, class)] = weights
                .row(class)
                .iter()
                .zip(inference_scaled.row(row).iter())
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                + bias[class];
        }
    }
    let probabilities = softmax_rows(&logits)?;
    Ok((scaler, class_weights, weights, bias, probabilities))
}

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, tolerance: f64) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: GPU {actual:.17e} drifted from CPU {expected:.17e}"
        );
    }
}

#[test]
fn full_gpu_fit_owns_original_label_mapping_counts_ddof0_scaling_and_pb_v2_training() -> Result<()>
{
    let (raw, original_labels) = raw_fixture();
    let (cpu_scaler, cpu_class_weights, cpu_weights, cpu_bias, _) =
        cpu_full_pipeline(&raw, &original_labels, &raw, 0.75, 3)?;

    let gpu = try_fit_passive_aggressive_cuda_full_pipeline(
        "gpu:0",
        "gpu:0",
        &raw,
        &original_labels,
        0.75,
        3,
    )?;

    assert_eq!(gpu.class_counts, [8, 8, 8]);
    assert_eq!(gpu.class_slack_weights, cpu_class_weights);
    assert_eq!(gpu.scaler_means.len(), raw.ncols());
    assert_eq!(gpu.scaler_stds.len(), raw.ncols());
    for (actual, expected) in gpu.scaler_means.iter().zip(cpu_scaler.means.iter()) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    for (actual, expected) in gpu.scaler_stds.iter().zip(cpu_scaler.stds.iter()) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(gpu.scaler_stds[2], 1.0, "constant columns use unit scale");
    assert_matrix_close(&gpu.weights, &cpu_weights, 1.0e-9);
    for (actual, expected) in gpu.bias.iter().zip(cpu_bias.iter()) {
        assert!((actual - expected).abs() <= 1.0e-9);
    }

    let pipeline = gpu
        .evidence
        .full_pipeline
        .as_ref()
        .expect("full-GPU fit must carry a full-pipeline receipt");
    assert_eq!(pipeline.requested_device_policy, "gpu:0");
    assert_eq!(pipeline.effective_device_policy, "gpu:0");
    assert_eq!(pipeline.raw_feature_h2d_bytes, (raw.len() * 8) as u64);
    assert_eq!(
        pipeline.original_label_h2d_bytes,
        (original_labels.len() * 4) as u64
    );
    assert_eq!(pipeline.scaled_feature_h2d_bytes, 0);
    assert_eq!(pipeline.remapped_label_h2d_bytes, 0);
    assert_eq!(pipeline.class_slack_weight_h2d_bytes, 0);
    assert_eq!(pipeline.parameter_initialization_h2d_bytes, 0);
    assert_eq!(pipeline.training_rows_per_launch, 1_024);
    assert_eq!(pipeline.training_row_chunk_count_per_epoch, 1);
    assert_eq!(pipeline.training_epoch_count, 3);
    assert_eq!(pipeline.training_launch_count, 3);
    assert_eq!(pipeline.training_interchunk_device_to_host_bytes, 0);
    assert_eq!(pipeline.kernel_launch_count, 12);
    assert_eq!(
        gpu.evidence.evidence_scope_schema,
        "neoethos.online_pa.cuda_evidence.whole_fit_call.v3"
    );
    assert_eq!(
        gpu.evidence.host_to_device_bytes,
        pipeline.raw_feature_h2d_bytes + pipeline.original_label_h2d_bytes
    );
    assert_eq!(
        gpu.evidence.device_to_host_bytes,
        pipeline.artifact_d2h_bytes
    );
    assert_eq!(pipeline.loss_cost_policy, "rho(y,r)=1; PA-I cap=C*w_y");
    assert_eq!(pipeline.residency_scope, "call_scoped");
    assert!(!pipeline.persistent_model_buffers);
    assert!(pipeline.device_identity.name.len() > 1);
    assert_eq!(pipeline.device_identity.ordinal, 0);
    Ok(())
}

#[test]
fn full_gpu_class_slack_weights_are_device_derived_clipped_and_require_all_classes() -> Result<()> {
    let raw = Array2::from_shape_fn((12, 2), |(row, col)| (row * 2 + col) as f64 / 7.0);
    let labels = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1];
    let fit =
        try_fit_passive_aggressive_cuda_full_pipeline("gpu:0", "gpu:0", &raw, &labels, 1.0, 1)?;
    assert_eq!(fit.class_counts, [10, 1, 1]);
    assert_eq!(fit.class_slack_weights, [0.5, 4.0, 4.0]);

    let invalid = vec![0, 1, -1, 7];
    let invalid_raw = Array2::zeros((invalid.len(), 2));
    let error = try_fit_passive_aggressive_cuda_full_pipeline(
        "gpu:0",
        "gpu:0",
        &invalid_raw,
        &invalid,
        1.0,
        1,
    )
    .expect_err("original-label validation must fail on the CUDA lane");
    assert!(format!("{error:#}").contains("label-map device fault"));

    let absent = vec![0, 1, 0, 1];
    let absent_raw = Array2::zeros((absent.len(), 2));
    let error = try_fit_passive_aggressive_cuda_full_pipeline(
        "gpu:0",
        "gpu:0",
        &absent_raw,
        &absent,
        1.0,
        1,
    )
    .expect_err("the GPU must not fabricate an absent class");
    assert!(format!("{error:#}").contains("all three classes"));
    Ok(())
}

#[test]
fn full_gpu_scaler_fails_closed_when_finite_inputs_overflow_ddof0_math() {
    let raw = Array2::from_shape_vec((3, 1), vec![f64::MAX, -f64::MAX, 0.0]).unwrap();
    let labels = vec![0, 1, -1];
    let error =
        try_fit_passive_aggressive_cuda_full_pipeline("gpu:0", "gpu:0", &raw, &labels, 1.0, 1)
            .expect_err("finite scaler input whose variance overflows must fail closed");
    assert!(format!("{error:#}").contains("scaler device fault"));
}

#[test]
fn fused_raw_gpu_inference_scales_logits_and_softmaxes_without_cpu_fallback() -> Result<()> {
    let (raw, labels) = raw_fixture();
    let inference_raw = raw.slice(ndarray::s![0..7, ..]).to_owned();
    let (_, _, _, _, cpu_probabilities) = cpu_full_pipeline(&raw, &labels, &inference_raw, 1.0, 3)?;
    let fit =
        try_fit_passive_aggressive_cuda_full_pipeline("gpu:0", "gpu:0", &raw, &labels, 1.0, 3)?;
    let prediction = try_predict_passive_aggressive_cuda_full_pipeline(
        "gpu:0",
        "gpu:0",
        &fit.device_identity,
        &inference_raw,
        &fit.scaler_means,
        &fit.scaler_stds,
        &fit.weights,
        &fit.bias,
    )?;

    assert_matrix_close(&prediction.probabilities, &cpu_probabilities, 1.0e-9);
    for row in prediction.probabilities.rows() {
        let sum = row.iter().sum::<f64>();
        assert!(row.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!((sum - 1.0).abs() <= 1.0e-12);
    }
    assert_eq!(prediction.evidence.kernel_launch_count, 1);
    assert_eq!(
        prediction.evidence.evidence_scope_schema,
        "neoethos.online_pa.cuda_evidence.whole_predict_call.v2"
    );
    assert_eq!(
        prediction.evidence.raw_feature_h2d_bytes,
        (inference_raw.len() * 8) as u64
    );
    assert_eq!(
        prediction.evidence.probability_d2h_bytes,
        (inference_raw.nrows() * 3 * 8) as u64
    );
    assert_eq!(
        prediction.evidence.host_to_device_bytes,
        prediction.evidence.raw_feature_h2d_bytes
            + prediction.evidence.scaler_parameter_h2d_bytes
            + prediction.evidence.model_parameter_h2d_bytes
    );
    assert_eq!(
        prediction.evidence.device_to_host_bytes,
        prediction.evidence.probability_d2h_bytes + prediction.evidence.status_d2h_bytes
    );
    assert_eq!(prediction.evidence.residency_scope, "call_scoped");
    assert!(!prediction.evidence.persistent_model_buffers);
    assert_eq!(
        prediction.runtime_backend,
        "online_pa_cuda_fused_inference[cuda:0]"
    );
    Ok(())
}

#[test]
fn fused_inference_rejects_overflowed_shifted_logits_before_exp() -> Result<()> {
    let (fit_raw, labels) = raw_fixture();
    let fit =
        try_fit_passive_aggressive_cuda_full_pipeline("gpu:0", "gpu:0", &fit_raw, &labels, 1.0, 1)?;
    let raw = Array2::zeros((1, 3));
    let scaler_means = vec![0.0; 3];
    let scaler_stds = vec![1.0; 3];
    let weights = Array2::zeros((3, 3));
    let bias = Array1::from_vec(vec![f64::MAX, -f64::MAX, 0.0]);
    let error = try_predict_passive_aggressive_cuda_full_pipeline(
        "gpu:0",
        "gpu:0",
        &fit.device_identity,
        &raw,
        &scaler_means,
        &scaler_stds,
        &weights,
        &bias,
    )
    .expect_err("an overflowed shifted logit must fault before exponentiation");
    assert_eq!(DEVICE_INFERENCE_ARITHMETIC_FAULT, 50);
    assert!(
        format!("{error:#}").contains("device fault code 50"),
        "unexpected inference overflow error: {error:#}"
    );
    Ok(())
}

#[test]
fn full_gpu_requested_effective_and_physical_identity_are_exact_and_mutation_closed() -> Result<()>
{
    let (raw, labels) = raw_fixture();
    for (requested, effective) in [("cpu", "cpu"), ("auto", "auto"), ("gpu:0", "gpu:1")] {
        assert!(
            try_fit_passive_aggressive_cuda_full_pipeline(
                requested, effective, &raw, &labels, 1.0, 1,
            )
            .is_err(),
            "{requested}/{effective} must not alias a CUDA ordinal"
        );
    }

    let fit =
        try_fit_passive_aggressive_cuda_full_pipeline("gpu:0", "gpu:0", &raw, &labels, 1.0, 1)?;
    let mut wrong_identity = fit.device_identity.clone();
    wrong_identity.uuid[0] ^= 0xff;
    let error = try_predict_passive_aggressive_cuda_full_pipeline(
        "gpu:0",
        "gpu:0",
        &wrong_identity,
        &raw,
        &fit.scaler_means,
        &fit.scaler_stds,
        &fit.weights,
        &fit.bias,
    )
    .expect_err("a different physical CUDA identity must fail closed");
    assert!(format!("{error:#}").contains("physical device identity"));
    Ok(())
}

#[test]
fn full_gpu_pipeline_preserves_prediction_based_correct_argmax_skip() -> Result<()> {
    // With zero parameters, CubeCL's documented tie handling selects class 2.
    // A class-2 target must therefore skip the first update under PB-v2.
    let raw = Array2::from_shape_vec((3, 1), vec![2.0, 2.0, 2.0])?;
    let labels = vec![-1, 0, 1];
    let fit =
        try_fit_passive_aggressive_cuda_full_pipeline("gpu:0", "gpu:0", &raw, &labels, 1.0, 1)?;
    let scaler = FeatureScaler {
        means: fit.scaler_means.clone(),
        stds: fit.scaler_stds.clone(),
    };
    let mapped = remap_three_class_labels(&labels)?;
    let scaled = scaler.transform(&raw)?;
    let (expected_weights, expected_bias) = prediction_based_weighted_slack_v2_cpu_oracle(
        &scaled,
        &mapped,
        &fit.class_slack_weights,
        1.0,
        1,
    );
    assert_matrix_close(&fit.weights, &expected_weights, 1.0e-12);
    assert_eq!(fit.bias, expected_bias);
    Ok(())
}

#[test]
fn epoch_major_training_chunks_preserve_exact_order_at_1024_row_boundaries() -> Result<()> {
    const EPOCHS: usize = 3;
    for rows in [1_023_usize, 1_024, 1_025] {
        let raw = Array2::from_shape_fn((rows, 8), |(row, col)| {
            let signal = if row % 3 == col % 3 { 1.75 } else { -0.875 };
            signal + ((row * 29 + col * 17) % 997) as f64 / 9970.0
        });
        let original_labels = (0..rows)
            .map(|row| match row % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect::<Vec<_>>();
        let mapped = remap_three_class_labels(&original_labels)?;
        let class_weights = clamped_balanced_class_slack_weights_v1(&mapped)?;
        let scaler = chunked_ddof0_scaler_cpu_oracle(&raw)?;
        let scaled = scaler.transform(&raw)?;
        let (expected_weights, expected_bias) = prediction_based_weighted_slack_v2_cpu_oracle(
            &scaled,
            &mapped,
            &class_weights,
            0.75,
            EPOCHS,
        );

        let gpu = try_fit_passive_aggressive_cuda_full_pipeline(
            "gpu:0",
            "gpu:0",
            &raw,
            &original_labels,
            0.75,
            EPOCHS,
        )?;
        assert_matrix_close(&gpu.weights, &expected_weights, 1.0e-9);
        for (actual, expected) in gpu.bias.iter().zip(expected_bias.iter()) {
            assert!((actual - expected).abs() <= 1.0e-9);
        }
        let receipt = gpu
            .evidence
            .full_pipeline
            .as_ref()
            .expect("chunked fit receipt");
        let expected_chunks = rows.div_ceil(1_024) as u64;
        assert_eq!(receipt.training_rows_per_launch, 1_024);
        assert_eq!(receipt.training_row_chunk_count_per_epoch, expected_chunks);
        assert_eq!(receipt.training_epoch_count, EPOCHS as u64);
        assert_eq!(
            receipt.training_launch_count,
            expected_chunks * EPOCHS as u64
        );
        assert_eq!(receipt.training_interchunk_device_to_host_bytes, 0);
        assert_eq!(
            receipt.kernel_launch_count,
            9 + receipt.training_launch_count
        );
        assert_eq!(
            gpu.evidence.kernel_launch_count,
            receipt.kernel_launch_count
        );
    }
    Ok(())
}

fn chunked_ddof0_scaler_cpu_oracle(raw: &Array2<f64>) -> Result<FeatureScaler> {
    const ROWS_PER_PARTIAL: usize = 1024;
    let mut means = Vec::with_capacity(raw.ncols());
    let mut stds = Vec::with_capacity(raw.ncols());
    for col in 0..raw.ncols() {
        let mut count = 0_u64;
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for chunk_start in (0..raw.nrows()).step_by(ROWS_PER_PARTIAL) {
            let chunk_end = (chunk_start + ROWS_PER_PARTIAL).min(raw.nrows());
            let mut partial_count = 0_u64;
            let mut partial_mean = 0.0_f64;
            let mut partial_m2 = 0.0_f64;
            for row in chunk_start..chunk_end {
                let value = raw[(row, col)];
                anyhow::ensure!(value.is_finite(), "oracle received non-finite input");
                partial_count += 1;
                let delta = value - partial_mean;
                partial_mean += delta / partial_count as f64;
                partial_m2 += delta * (value - partial_mean);
                anyhow::ensure!(
                    partial_mean.is_finite() && partial_m2.is_finite(),
                    "oracle scaler partial overflow"
                );
            }
            let combined_count = count + partial_count;
            let delta = partial_mean - mean;
            let next_mean = mean + delta * partial_count as f64 / combined_count as f64;
            let count_product = count as f64 * partial_count as f64;
            let cross = delta * delta * count_product / combined_count as f64;
            let next_m2 = m2 + partial_m2 + cross;
            anyhow::ensure!(
                next_mean.is_finite() && next_m2.is_finite(),
                "oracle scaler finalize overflow"
            );
            count = combined_count;
            mean = next_mean;
            m2 = next_m2;
        }
        let std = (m2 / count as f64).sqrt();
        anyhow::ensure!(std.is_finite(), "oracle scaler output overflow");
        means.push(mean);
        stds.push(if std > 1.0e-12 { std } else { 1.0 });
    }
    Ok(FeatureScaler { means, stds })
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[test]
fn same_workload_full_numerical_gpu_pipeline_beats_cpu_including_transfers_and_sync() -> Result<()>
{
    let warm = raw_fixture();
    let warm_fit =
        try_fit_passive_aggressive_cuda_full_pipeline("gpu:0", "gpu:0", &warm.0, &warm.1, 1.0, 1)?;
    try_predict_passive_aggressive_cuda_full_pipeline(
        "gpu:0",
        "gpu:0",
        &warm_fit.device_identity,
        &warm.0,
        &warm_fit.scaler_means,
        &warm_fit.scaler_stds,
        &warm_fit.weights,
        &warm_fit.bias,
    )?;

    let rows = 192;
    let cols = 4096;
    let raw = Array2::from_shape_fn((rows, cols), |(row, col)| {
        (((row * 131 + col * 17) % 2048) as f64 - 1024.0) / 1024.0
    });
    let labels = (0..rows)
        .map(|row| match row % 3 {
            0 => 0,
            1 => 1,
            _ => -1,
        })
        .collect::<Vec<_>>();

    let mut cpu_samples = Vec::new();
    let mut gpu_samples = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let _ = cpu_full_pipeline(&raw, &labels, &raw, 1.0, 2)?;
        cpu_samples.push(started.elapsed());

        let started = Instant::now();
        let fit =
            try_fit_passive_aggressive_cuda_full_pipeline("gpu:0", "gpu:0", &raw, &labels, 1.0, 2)?;
        let prediction = try_predict_passive_aggressive_cuda_full_pipeline(
            "gpu:0",
            "gpu:0",
            &fit.device_identity,
            &raw,
            &fit.scaler_means,
            &fit.scaler_stds,
            &fit.weights,
            &fit.bias,
        )?;
        assert_eq!(prediction.probabilities.dim(), (rows, 3));
        gpu_samples.push(started.elapsed());
    }

    let cpu = median(cpu_samples);
    let gpu = median(gpu_samples);
    eprintln!(
        "ONLINE_PA_FULL_GPU_BENCH rows={rows} cols={cols} epochs=2 cpu_median_us={} gpu_median_us={} speedup={:.3}",
        cpu.as_micros(),
        gpu.as_micros(),
        cpu.as_secs_f64() / gpu.as_secs_f64(),
    );
    assert!(
        gpu.as_secs_f64() * 1.25 < cpu.as_secs_f64(),
        "full GPU {:?} must be at least 1.25x faster than full CPU {:?}",
        gpu,
        cpu
    );
    Ok(())
}
