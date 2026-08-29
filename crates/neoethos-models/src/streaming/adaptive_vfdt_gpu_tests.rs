use ndarray::Array2;

use super::adaptive_gpu::{try_fit_hoeffding_vfdt_cuda, try_predict_hoeffding_vfdt_cuda};

#[test]
fn hoeffding_vfdt_training_and_inference_execute_on_cuda_without_linear_fallback() {
    let features = Array2::from_shape_fn((90, 2), |(row, col)| {
        let class = row / 30;
        if col == 0 {
            class as f64 * 3.0 + (row % 5) as f64 * 0.05
        } else {
            ((row * 7 % 17) as f64 - 8.0) / 8.0
        }
    });
    let labels = (0..90).map(|row| row / 30).collect::<Vec<_>>();

    let fit = try_fit_hoeffding_vfdt_cuda("cuda:0", &features, &labels, 16, 1.0e-7, 0.05, 12)
        .expect("Hoeffding/VFDT sufficient statistics and split should execute on CUDA");

    assert_eq!(fit.schema, "neoethos.online_hoeffding.cuda_vfdt.f64.v1");
    assert_eq!(fit.runtime_backend, "online_hoeffding_vfdt_cuda[cuda:0]");
    assert_eq!(fit.effective_device_policy, "gpu:0");
    assert!(fit.root.split_feature_index.is_some());
    assert!(fit.root.split_threshold.is_some());
    assert!(fit.evidence.kernel_launch_count >= 3);
    assert!(fit.evidence.host_to_device_bytes > 0);
    assert!(fit.evidence.device_to_host_bytes > 0);

    let probabilities = try_predict_hoeffding_vfdt_cuda("cuda:0", &features, &fit)
        .expect("verified VFDT artifact should infer on CUDA");
    assert_eq!(probabilities.dim(), (90, 3));
    for row in probabilities.outer_iter() {
        assert!(row.iter().all(|value| value.is_finite()));
        assert!((row.sum() - 1.0).abs() <= 1.0e-9);
    }
}
