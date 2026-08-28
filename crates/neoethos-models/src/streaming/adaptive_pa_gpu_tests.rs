use ndarray::{Array1, Array2};
use std::time::Instant;

use super::adaptive_gpu::try_fit_passive_aggressive_cuda;

fn fixture() -> (Array2<f64>, Vec<usize>, [f64; 3]) {
    let features = Array2::from_shape_fn((15, 2), |(row, col)| {
        if col == 0 {
            row as f64 / 7.0 - 1.0
        } else {
            ((row * 3 % 7) as f64 - 3.0) / 3.0
        }
    });
    let labels = (0..15).map(|row| row % 3).collect::<Vec<_>>();
    (features, labels, [1.0, 1.25, 0.75])
}

fn ordered_cpu_oracle(
    features: &Array2<f64>,
    labels: &[usize],
    class_weights: &[f64; 3],
    aggressiveness: f64,
    epochs: usize,
) -> (Array2<f64>, Array1<f64>) {
    let mut weights = Array2::<f64>::zeros((3, features.ncols()));
    let mut bias = Array1::<f64>::zeros(3);
    for _ in 0..epochs {
        for (row, target_class) in labels.iter().copied().enumerate() {
            let x_row = features.row(row);
            // The class prototypes each own a bias.  In the paper's joint
            // feature space the update direction is therefore
            // (x, 1) for the target and -(x, 1) for the predicted class,
            // whose squared norm is 2 * (||x||^2 + 1).
            let augmented_norm_sq = x_row.iter().map(|value| value * value).sum::<f64>() + 1.0;
            let mut scores = [0.0; 3];
            for class_idx in 0..3 {
                scores[class_idx] = weights
                    .row(class_idx)
                    .iter()
                    .zip(x_row.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>()
                    + bias[class_idx];
            }
            if scores.iter().any(|score| !score.is_finite()) {
                continue;
            }
            let predicted_class = scores
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(class_idx, _)| class_idx)
                .unwrap_or(target_class);
            // Crammer et al. JMLR 2006 section 8, equations 45 and 47-49:
            // the prediction-based loss is zero for a correct argmax.
            if predicted_class == target_class {
                continue;
            }
            let margin = scores[predicted_class] - scores[target_class] + 1.0;
            if margin <= 0.0 {
                continue;
            }
            // The class weight extends PA-I's slack penalty, so it scales the
            // cap C*w_y rather than the prediction-based loss numerator.
            let tau = (margin / (2.0 * augmented_norm_sq))
                .min(aggressiveness * class_weights[target_class]);
            if !tau.is_finite() || tau < 0.0 {
                continue;
            }
            for col in 0..features.ncols() {
                weights[(target_class, col)] += tau * x_row[col];
                weights[(predicted_class, col)] -= tau * x_row[col];
            }
            bias[target_class] += tau;
            bias[predicted_class] -= tau;
        }
    }
    (weights, bias)
}

#[test]
fn passive_aggressive_cuda_matches_prediction_based_weighted_slack_v2_oracle() {
    let (features, labels, class_weights) = fixture();
    let (expected_weights, expected_bias) =
        ordered_cpu_oracle(&features, &labels, &class_weights, 1.0, 2);

    let fit = try_fit_passive_aggressive_cuda("cuda:0", &features, &labels, &class_weights, 1.0, 2)
        .expect("passive-aggressive updates should execute on CUDA");

    assert_eq!(fit.runtime_backend, "online_pa_cuda[cuda:0]");
    assert_eq!(fit.effective_device_policy, "gpu:0");
    assert_eq!(
        fit.training_semantics_schema,
        "neoethos.online_pa.prediction_based.class_weighted_slack_cap.bias_augmented.f64.v2"
    );
    assert_eq!(fit.weights.dim(), (3, 2));
    assert_eq!(fit.bias.len(), 3);
    assert_eq!(fit.evidence.kernel_launch_count, 2);
    assert_eq!(fit.evidence.host_to_device_bytes, 400);
    assert_eq!(fit.evidence.device_to_host_bytes, 76);
    assert_eq!(fit.evidence.ordered_sample_visits, 30);

    for (actual, expected) in fit.weights.iter().zip(expected_weights.iter()) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "CUDA weight {actual:.17e} drifted from ordered oracle {expected:.17e}"
        );
    }
    for (actual, expected) in fit.bias.iter().zip(expected_bias.iter()) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "CUDA bias {actual:.17e} drifted from ordered oracle {expected:.17e}"
        );
    }
    assert!(fit.weights.iter().any(|value| *value != 0.0));
}

#[test]
fn passive_aggressive_cuda_rejects_every_non_cuda_policy_without_fallback() {
    let (features, labels, class_weights) = fixture();
    for policy in ["cpu", "auto", "vulkan:0", "rocm:0", "bogus"] {
        let error =
            try_fit_passive_aggressive_cuda(policy, &features, &labels, &class_weights, 1.0, 1)
                .expect_err("a non-CUDA policy must fail closed");
        assert!(
            error.to_string().contains("CUDA"),
            "unexpected rejection for {policy}: {error:#}"
        );
    }
}

#[test]
fn passive_aggressive_cuda_rejects_malformed_work_without_launching() {
    let (features, mut labels, class_weights) = fixture();
    labels[0] = 3;
    assert!(
        try_fit_passive_aggressive_cuda("cuda:0", &features, &labels, &class_weights, 1.0, 2,)
            .is_err()
    );
    let (_, labels, _) = fixture();
    assert!(
        try_fit_passive_aggressive_cuda(
            "cuda:0",
            &features,
            &labels,
            &[1.0, f64::NAN, 1.0],
            1.0,
            2,
        )
        .is_err()
    );
    for class_weights in [[0.49, 1.0, 1.0], [4.01, 1.0, 1.0]] {
        assert!(
            try_fit_passive_aggressive_cuda("cuda:0", &features, &labels, &class_weights, 1.0, 2,)
                .is_err()
        );
    }
    assert!(
        try_fit_passive_aggressive_cuda("cuda:0", &features, &labels, &[1.0; 3], f64::INFINITY, 2,)
            .is_err()
    );
}

#[test]
fn passive_aggressive_cuda_refuses_finite_input_that_overflows_device_math() {
    let features = Array2::from_shape_vec((1, 2), vec![f64::MAX, f64::MAX])
        .expect("overflow fixture has the declared shape");
    let labels = vec![0];
    let error = try_fit_passive_aggressive_cuda("cuda:0", &features, &labels, &[1.0; 3], 1.0, 1)
        .expect_err("finite inputs that overflow device arithmetic must fail closed");
    assert!(
        format!("{error:#}").contains("device arithmetic fault"),
        "unexpected overflow rejection: {error:#}"
    );
}

#[test]
fn passive_aggressive_block_cooperative_cuda_beats_the_same_ordered_cpu_workload() {
    // Compile/JIT the exact specialization before timing either implementation.
    let (warm_features, warm_labels, warm_weights) = fixture();
    try_fit_passive_aggressive_cuda(
        "cuda:0",
        &warm_features,
        &warm_labels,
        &warm_weights,
        1.0,
        1,
    )
    .expect("warm the ordered PA CUDA specialization");

    let rows = 192;
    let cols = 4096;
    let features = Array2::from_shape_fn((rows, cols), |(row, col)| {
        let centered = ((row * 131 + col * 17) % 2048) as f64 - 1024.0;
        centered / 1024.0
    });
    let labels = (0..rows).map(|row| row % 3).collect::<Vec<_>>();
    let class_weights = [1.0, 1.125, 0.875];

    let cpu_started = Instant::now();
    let (cpu_weights, cpu_bias) = ordered_cpu_oracle(&features, &labels, &class_weights, 1.0, 2);
    let cpu_elapsed = cpu_started.elapsed();

    let gpu_started = Instant::now();
    let gpu = try_fit_passive_aggressive_cuda("cuda:0", &features, &labels, &class_weights, 1.0, 2)
        .expect("benchmark the same ordered PA workload on CUDA");
    let gpu_elapsed = gpu_started.elapsed();

    let max_weight_drift = gpu
        .weights
        .iter()
        .zip(cpu_weights.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    let max_bias_drift = gpu
        .bias
        .iter()
        .zip(cpu_bias.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "ONLINE_PA_CUDA_BENCH rows={rows} cols={cols} epochs=2 cpu_us={} gpu_us={} max_weight_drift={max_weight_drift:.3e} max_bias_drift={max_bias_drift:.3e} launches={} h2d={} d2h={}",
        cpu_elapsed.as_micros(),
        gpu_elapsed.as_micros(),
        gpu.evidence.kernel_launch_count,
        gpu.evidence.host_to_device_bytes,
        gpu.evidence.device_to_host_bytes,
    );
    assert!(
        max_weight_drift <= 1.0e-9,
        "weight drift {max_weight_drift}"
    );
    assert!(max_bias_drift <= 1.0e-9, "bias drift {max_bias_drift}");
    assert!(
        gpu_elapsed < cpu_elapsed,
        "block-cooperative CUDA {:?} must beat ordered CPU {:?}",
        gpu_elapsed,
        cpu_elapsed
    );
}
