use ndarray::Array2;

use super::bayesian_gpu::try_fit_bayesian_ovr_cuda;

#[test]
fn bayesian_ovr_training_executes_f64_cuda_and_returns_full_posteriors() {
    let features = Array2::from_shape_fn((18, 3), |(row, col)| match col {
        0 => (row as f64 - 8.5) / 4.0,
        1 => ((row * 5 % 11) as f64 - 5.0) / 3.0,
        _ => ((row * row % 13) as f64 - 6.0) / 5.0,
    });
    let labels = (0..18).map(|row| row % 3).collect::<Vec<_>>();

    let fit = try_fit_bayesian_ovr_cuda(
        "cuda:0",
        &features,
        &labels,
        None,
        &[],
        0.05,
        0.05,
        3,
    )
    .expect("Bayesian OVR should train through the selected CUDA device");

    assert_eq!(fit.runtime_backend, "bayes_logit_bayesian_ovr_cuda[cuda:0]");
    assert_eq!(fit.effective_device_policy, "gpu:0");
    assert_eq!(fit.classes.len(), 3);
    assert!(fit.evidence.kernel_launch_count >= 3);
    assert!(fit.evidence.host_to_device_bytes > 0);
    assert!(fit.evidence.device_to_host_bytes > 0);
    for class in fit.classes {
        assert_eq!(class.weights.len(), 3);
        assert_eq!(class.covariance.dim(), (4, 4));
        assert!(class.weights.iter().all(|value| value.is_finite()));
        assert!(class.bias.is_finite());
        assert!(class.covariance.iter().all(|value| value.is_finite()));
        assert!((0..4).all(|idx| class.covariance[(idx, idx)] > 0.0));
    }
}
