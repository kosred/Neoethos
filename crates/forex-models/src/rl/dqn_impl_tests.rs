use super::*;


use ndarray::{Array1, Array2};
use polars::prelude::NamedFrom;
use std::path::PathBuf;

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{stamp}-{}", std::process::id()));
    std::fs::create_dir_all(&path).expect("create temp directory");
    path
}

#[test]
fn rollback_rl_target_removes_partial_file_without_backup() {
    let path = unique_temp_dir("rl-rollback-target");
    let final_path = path.join("partial.json");
    let backup_path = path.join("partial.json.bak");
    std::fs::write(&final_path, b"partial").expect("write partial artifact");

    rollback_rl_target(&final_path, &backup_path);

    assert!(
        !final_path.exists(),
        "rollback should remove partial final artifact when no backup exists"
    );
    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn fallback_only_artifact_round_trips_without_network_file() {
    let mut learner = TradingReinforcementLearner::new();
    learner.train_args.state_dim = 2;
    learner.train_args.train_rows = 16;
    learner.train_args.feature_columns = vec!["f1".to_string(), "f2".to_string()];
    learner.train_args.hidden_dims = vec![4, 4];
    learner.train_args.state_mins = vec![0.0, 0.0];
    learner.train_args.state_maxs = vec![1.0, 1.0];
    learner.bounds = Some(FeatureBounds {
        mins: vec![0.0, 0.0],
        maxs: vec![1.0, 1.0],
    });
    learner.feature_columns = learner.train_args.feature_columns.clone();
    learner.train_args.backend = "linear_q_cpu".to_string();
    learner.train_args.device_policy = "cpu".to_string();
    learner.train_args.requested_backend = Some("linear_q_cpu".to_string());
    learner.train_args.requested_device_policy = Some("cpu".to_string());
    learner.train_args.effective_backend = Some("linear_q_cpu".to_string());
    learner.train_args.effective_device_policy = Some("cpu".to_string());

    let weights = Array2::<f32>::from_shape_vec((3, 2), vec![1.0_f32, 0.0, 0.0, 1.0, 0.5, 0.5])
        .expect("shape fallback weights");
    let bias = Array1::<f32>::from_vec(vec![0.1_f32, 0.2, -0.1]);
    learner.train_args.fallback_weights = Some(weights.clone());
    learner.train_args.fallback_bias = Some(bias.clone());
    learner.train_args.training_report = Some(TradingRlTrainingReport {
        train_rows: 16,
        episode_count: 2,
        state_dim: 2,
        reward_horizon: 0,
        episode_len: 0,
        backend: "linear_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        average_hold_reward: 0.1,
        average_buy_reward: 0.2,
        average_sell_reward: -0.1,
        used_network_snapshot: false,
        used_fallback_q: true,
        used_feature_scaler: false,
    });
    learner.fallback_weights = Some(weights);
    learner.fallback_bias = Some(bias);
    learner.training_report = learner.train_args.training_report.clone();

    let path = unique_temp_dir("rl-fallback-only");
    learner
        .save(&path)
        .expect("save should succeed without a network file");

    assert!(
        path.join("rl_config.json").exists(),
        "artifact config should be written"
    );
    assert!(
        !path.join("q_network.safetensors").exists(),
        "fallback-only save should not create a network snapshot"
    );

    let loaded = TradingReinforcementLearner::load(&path)
        .expect("load should accept a fallback-only artifact");
    let q_values = loaded
        .predict_q_values(&[0.25_f32, 0.75_f32])
        .expect("fallback inference should work after load");

    assert_eq!(q_values.len(), 3);
    assert!((q_values[0] - 0.35).abs() < 1e-6);
    assert!((q_values[1] - 0.95).abs() < 1e-6);
    assert!((q_values[2] - 0.4).abs() < 1e-6);
    assert_eq!(
        loaded
            .training_report
            .as_ref()
            .expect("training report should round-trip")
            .backend,
        "linear_q_cpu"
    );

    let _ = std::fs::remove_dir_all(&path);
}

#[cfg(feature = "reinforcement-learning")]
#[test]
fn runtime_hints_are_normalized() {
    let learner =
        TradingReinforcementLearner::new().with_runtime_hints("RLKIT", "CUDA:0", 2, 4, 0, 1);

    assert_eq!(learner.train_args.backend, "rlkit");
    assert_eq!(learner.train_args.device_policy, "cuda:0");
}

#[cfg(feature = "reinforcement-learning")]
#[test]
fn auto_policy_uses_cpu_when_cuda_backend_is_unavailable() {
    let (device, effective_policy, effective_backend) =
        resolve_rl_training_device("auto").expect("auto policy should resolve");

    #[cfg(not(feature = "reinforcement-learning-cuda"))]
    {
        assert!(matches!(device, Device::Cpu));
        assert_eq!(effective_policy, "cpu");
        assert_eq!(effective_backend, "rlkit_cpu");
    }

    #[cfg(feature = "reinforcement-learning-cuda")]
    {
        let _ = device;
        assert!(
            matches!(effective_policy.as_str(), "cpu") || effective_policy.starts_with("cuda:")
        );
        assert!(matches!(
            effective_backend.as_str(),
            "rlkit_cpu" | "rlkit_cuda"
        ));
    }
}

#[cfg(all(
    feature = "reinforcement-learning",
    not(feature = "reinforcement-learning-cuda")
))]
#[test]
fn explicit_gpu_policy_falls_back_to_cpu_without_cuda_support() {
    let (device, policy, backend) =
        resolve_rl_training_device("rocm:0").expect("gpu policy should degrade to cpu");
    assert!(matches!(device, Device::Cpu));
    assert_eq!(policy, "cpu");
    assert_eq!(backend, "rlkit_cpu");
}

#[test]
fn normalize_rl_device_policy_accepts_vendor_neutral_gpu_tokens() {
    assert_eq!(normalize_rl_device_policy("CUDA:1"), "gpu:1");
    assert_eq!(normalize_rl_device_policy("rocm:2"), "gpu:2");
    assert_eq!(normalize_rl_device_policy("metal:0"), "gpu:0");
    assert_eq!(normalize_rl_device_policy("vulkan:3"), "gpu:3");
    assert_eq!(normalize_rl_device_policy("nvidia"), "gpu");
}

#[test]
fn validate_artifact_rejects_partial_fallback_parameters() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 64,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("linear_q_cpu".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("linear_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "linear_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 0,
        episode_len: 0,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "linear_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.0,
            average_buy_reward: 0.0,
            average_sell_reward: 0.0,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Linear,
        fallback_weights: Some(Array2::zeros((3, 2))),
        fallback_bias: None,
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("partial fallback parameters should be rejected");
    assert!(
        err.to_string().contains("persisted together"),
        "unexpected error: {err}"
    );
}

#[test]
fn validate_q_values_rejects_non_finite_rows() {
    let err = validate_q_values(vec![0.1, f32::NAN, 0.2])
        .expect_err("non-finite q-values should be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[test]
fn quadratic_fallback_basis_appends_squared_terms() {
    let expanded = expand_fallback_basis(&[0.5, -0.25], TradingFallbackBasis::Quadratic);
    assert_eq!(expanded, vec![0.5, -0.25, 0.25, 0.0625]);
}

#[test]
fn runtime_backend_details_reflect_quadratic_fallback_basis() {
    let mut learner = TradingReinforcementLearner::new();
    learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
    learner.train_args.fallback_basis = TradingFallbackBasis::Quadratic;
    learner.fallback_weights = Some(Array2::zeros((3, 4)));
    learner.fallback_bias = Some(Array1::zeros(3));

    let (backend, degraded_reason) = learner.runtime_backend_details();
    assert_eq!(backend.as_deref(), Some("quadratic_q_cpu"));
    assert_eq!(degraded_reason.as_deref(), Some("rl_network_unavailable"));
}

#[test]
fn runtime_backend_details_explain_requested_gpu_fallback_to_cpu() {
    let mut learner = TradingReinforcementLearner::new();
    learner.train_args.requested_backend = Some("rlkit".to_string());
    learner.train_args.requested_device_policy = Some("cuda:0".to_string());
    learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
    learner.train_args.effective_device_policy = Some("cpu".to_string());
    learner.train_args.fallback_basis = TradingFallbackBasis::Quadratic;
    learner.fallback_weights = Some(Array2::zeros((3, 4)));
    learner.fallback_bias = Some(Array1::zeros(3));

    let (backend, degraded_reason) = learner.runtime_backend_details();
    assert_eq!(backend.as_deref(), Some("quadratic_q_cpu"));
    let degraded_reason = degraded_reason.expect("fallback should be degraded");
    assert!(degraded_reason.contains("requested_rl_device_unavailable"));
    assert!(degraded_reason.contains("rl_network_unavailable"));
    assert!(degraded_reason.contains("rl_backend_degraded_to_fallback_q"));
}

#[test]
fn runtime_backend_details_explain_requested_precision_when_unavailable() {
    unsafe {
        std::env::set_var("FOREX_BOT_DQN_TRAIN_PRECISION", "bf16");
    }
    let learner = TradingReinforcementLearner::new();
    let (_backend, degraded_reason) = learner.runtime_backend_details();
    unsafe {
        std::env::remove_var("FOREX_BOT_DQN_TRAIN_PRECISION");
    }

    let degraded_reason =
        degraded_reason.expect("precision request should appear in degraded reason");
    assert!(degraded_reason.contains("requested_rl_precision_unavailable(bf16)"));
}

#[test]
fn rl_precision_resolution_uses_bf16_on_supported_cuda_runtime() {
    let (effective_precision, degraded_reason) = resolve_rl_training_precision_with_capability(
        Some("bf16"),
        "rlkit_cuda",
        "cuda:0",
        Some(true),
    );

    assert_eq!(effective_precision, "bf16");
    assert!(degraded_reason.is_none());
}

#[test]
fn rl_precision_resolution_uses_bf16_on_cpu_runtime() {
    let (effective_precision, degraded_reason) = resolve_rl_training_precision_with_capability(
        Some("bf16"),
        "rlkit_cpu",
        "cpu",
        Some(true),
    );

    assert_eq!(effective_precision, "bf16");
    assert!(degraded_reason.is_none());
}

#[test]
fn rl_precision_resolution_explains_cpu_backend_limit() {
    let (effective_precision, degraded_reason) = resolve_rl_training_precision_with_capability(
        Some("bf16"),
        "quadratic_q_cpu",
        "cpu",
        None,
    );

    assert_eq!(effective_precision, "fp32");
    let degraded_reason = degraded_reason.expect("bf16 request should degrade");
    assert!(degraded_reason.contains("requested_rl_precision_unavailable(bf16)"));
    assert!(degraded_reason.contains("rl_backend_precision_limit(quadratic_q_cpu->fp32)"));
}

#[test]
fn rl_precision_resolution_degrades_lower_precision_requests_to_bf16_when_available() {
    let (effective_precision, degraded_reason) = resolve_rl_training_precision_with_capability(
        Some("fp8"),
        "rlkit_cuda",
        "cuda:0",
        Some(true),
    );

    assert_eq!(effective_precision, "bf16");
    let degraded_reason = degraded_reason.expect("fp8 request should degrade");
    assert!(degraded_reason.contains("requested_rl_precision_unavailable(fp8)"));
    assert!(degraded_reason.contains("rl_precision_degraded_to_bf16"));
}

#[test]
fn validate_artifact_rejects_network_precision_on_fallback_backend() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 64,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("linear_q_cpu".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("linear_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: Some("bf16".to_string()),
        backend: "linear_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 0,
        episode_len: 0,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "linear_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.0,
            average_buy_reward: 0.0,
            average_sell_reward: 0.0,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Linear,
        fallback_weights: Some(Array2::zeros((3, 2))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("fallback backend should reject neural network precision metadata");
    assert!(err.to_string().contains("network_precision"));
}

#[test]
fn validate_artifact_rejects_training_report_claiming_missing_fallback() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 64,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("rlkit_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "rlkit_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 0,
        episode_len: 0,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "rlkit_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: None,
        fallback_bias: None,
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("training report should not claim fallback without fallback parameters");
    assert!(err.to_string().contains("fallback"));
}

#[test]
fn validate_artifact_rejects_training_report_underreporting_fallback_backend() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 64,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("quadratic_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "quadratic_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 0,
        episode_len: 0,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: false,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("fallback backend must not under-report fallback usage");
    assert!(err.to_string().contains("fallback Q as unused"));
}

#[test]
fn validate_artifact_rejects_training_report_claiming_network_on_fallback_backend() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 64,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("cuda:0".to_string()),
        effective_backend: Some("quadratic_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "quadratic_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 4,
        episode_len: 16,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 4,
            state_dim: 2,
            reward_horizon: 4,
            episode_len: 16,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: true,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("network-on-fallback report should be rejected");
    assert!(err.to_string().contains("network snapshot"));
}

#[test]
fn validate_artifact_rejects_missing_training_report() {
    let mut artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 64,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("linear_q_cpu".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("linear_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "linear_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 4,
        episode_len: 16,
        training_report: None,
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };
    artifact.training_report = None;

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("missing training_report should be rejected");
    assert!(err.to_string().contains("missing training_report"));
}

#[test]
fn validate_artifact_rejects_zero_training_parameters() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 0,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("auto".to_string()),
        effective_backend: Some("quadratic_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "quadratic_q_cpu".to_string(),
        device_policy: "auto".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 4,
        episode_len: 16,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 4,
            state_dim: 2,
            reward_horizon: 4,
            episode_len: 16,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("zero epochs must be rejected");
    assert!(err.to_string().contains("zero-valued training parameters"));
}

#[test]
fn runtime_backend_details_include_missing_training_report_for_trained_state() {
    let mut learner = TradingReinforcementLearner::default();
    learner.train_args.train_rows = 128;
    learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
    learner.train_args.effective_device_policy = Some("cpu".to_string());
    learner.train_args.fallback_basis = TradingFallbackBasis::Quadratic;
    learner.fallback_weights = Some(Array2::zeros((3, 4)));
    learner.fallback_bias = Some(Array1::zeros(3));
    learner.training_report = None;

    let (backend, degraded_reason) = learner.runtime_backend_details();

    assert_eq!(backend.as_deref(), Some("quadratic_q_cpu"));
    let degraded_reason = degraded_reason.expect("trained fallback state should be degraded");
    assert!(degraded_reason.contains("rl_training_report_missing"));
    assert!(degraded_reason.contains("rl_network_unavailable"));
}

#[test]
fn runtime_backend_details_flag_missing_persisted_training_report() {
    let mut learner = TradingReinforcementLearner::default();
    learner.train_args.train_rows = 128;
    learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
    learner.train_args.effective_device_policy = Some("cpu".to_string());
    learner.fallback_weights = Some(Array2::zeros((3, 2)));
    learner.fallback_bias = Some(Array1::zeros(3));
    learner.training_report = Some(TradingRlTrainingReport {
        train_rows: 128,
        episode_count: 4,
        state_dim: 2,
        reward_horizon: 4,
        episode_len: 16,
        backend: "quadratic_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        average_hold_reward: 0.1,
        average_buy_reward: 0.2,
        average_sell_reward: -0.1,
        used_network_snapshot: false,
        used_fallback_q: true,
        used_feature_scaler: false,
    });
    learner.train_args.training_report = None;

    let (_, degraded_reason) = learner.runtime_backend_details();
    assert!(
        degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("rl_persisted_training_report_missing")
    );
}

#[test]
fn validate_artifact_rejects_partial_runtime_identity_fields() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 8,
        max_steps: 128,
        update_interval: 8,
        update_freq: 2,
        batch_size: 32,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: None,
        effective_backend: Some("quadratic_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "rlkit".to_string(),
        device_policy: "auto".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 4,
        episode_len: 16,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 4,
            state_dim: 2,
            reward_horizon: 4,
            episode_len: 16,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("partial runtime identity should be rejected");
    assert!(err.to_string().contains("requested/effective backend"));
}

#[test]
fn validate_artifact_rejects_unknown_effective_backend() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 8,
        max_steps: 128,
        update_interval: 8,
        update_freq: 2,
        batch_size: 32,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("auto".to_string()),
        effective_backend: Some("mystery_backend".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "rlkit".to_string(),
        device_policy: "auto".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 4,
        episode_len: 16,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 4,
            state_dim: 2,
            reward_horizon: 4,
            episode_len: 16,
            backend: "mystery_backend".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("unknown effective backend should be rejected");
    assert!(err.to_string().contains("supported runtime backend"));
}

#[test]
fn validate_artifact_rejects_legacy_effective_runtime_drift() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 8,
        max_steps: 128,
        update_interval: 8,
        update_freq: 2,
        batch_size: 32,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("cuda:0".to_string()),
        effective_backend: Some("quadratic_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "rlkit_cuda".to_string(),
        device_policy: "cuda:0".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 4,
        episode_len: 16,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 4,
            state_dim: 2,
            reward_horizon: 4,
            episode_len: 16,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("legacy effective runtime drift should be rejected");
    assert!(err.to_string().contains("legacy backend"));
}

#[test]
fn runtime_backend_details_include_training_report_backend_drift() {
    let mut learner = TradingReinforcementLearner::default();
    learner.train_args.train_rows = 128;
    learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
    learner.train_args.effective_device_policy = Some("cpu".to_string());
    learner.fallback_weights = Some(Array2::zeros((3, 2)));
    learner.fallback_bias = Some(Array1::zeros(3));
    learner.training_report = Some(TradingRlTrainingReport {
        backend: "rlkit_cuda".to_string(),
        device_policy: "cpu".to_string(),
        used_fallback_q: true,
        used_feature_scaler: false,
        ..TradingRlTrainingReport::default()
    });

    let (_, degraded_reason) = learner.runtime_backend_details();
    assert!(
        degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("rl_training_report_backend_drift")
    );
}

#[test]
fn runtime_backend_details_include_persisted_runtime_drift_and_missing_network_snapshot() {
    let mut learner = TradingReinforcementLearner::default();
    learner.train_args.train_rows = 128;
    learner.train_args.requested_backend = Some("rlkit".to_string());
    learner.train_args.requested_device_policy = Some("cuda:0".to_string());
    learner.train_args.effective_backend = Some("rlkit_cuda".to_string());
    learner.train_args.effective_device_policy = Some("cuda:0".to_string());
    learner.runtime_effective_backend = Some("quadratic_q_cpu".to_string());
    learner.runtime_effective_device_policy = Some("cpu".to_string());
    learner.persisted_network_snapshot_present = true;
    learner.fallback_weights = Some(Array2::zeros((3, 2)));
    learner.fallback_bias = Some(Array1::zeros(3));

    let (_, degraded_reason) = learner.runtime_backend_details();
    let degraded_reason = degraded_reason.expect("runtime should be degraded");
    assert!(degraded_reason.contains("rl_persisted_runtime_backend_drift"));
    assert!(degraded_reason.contains("rl_persisted_runtime_device_drift"));
    assert!(degraded_reason.contains("persisted_rl_network_snapshot_unavailable"));
}

#[test]
fn validate_artifact_rejects_misaligned_feature_scaler() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 8,
        max_steps: 128,
        update_interval: 8,
        update_freq: 2,
        batch_size: 32,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("quadratic_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "quadratic_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 4,
        episode_len: 16,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 4,
            state_dim: 2,
            reward_horizon: 4,
            episode_len: 16,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: true,
        }),
        feature_scaler: Some(FeatureScaler {
            means: vec![0.0],
            stds: vec![1.0, 1.0],
        }),
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("misaligned feature scaler should be rejected");
    assert!(err.to_string().contains("feature_scaler mismatch"));
}

#[test]
fn validate_artifact_rejects_training_report_scaler_drift() {
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 8,
        max_steps: 128,
        update_interval: 8,
        update_freq: 2,
        batch_size: 32,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("quadratic_q_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "quadratic_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 4,
        episode_len: 16,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 4,
            state_dim: 2,
            reward_horizon: 4,
            episode_len: 16,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: Some(FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        }),
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    let err = TradingReinforcementLearner::validate_artifact(&artifact)
        .expect_err("training report scaler drift should be rejected");
    assert!(err.to_string().contains("feature_scaler flag"));
}

#[test]
fn artifact_rejects_live_training_report_drift() {
    let mut learner = TradingReinforcementLearner::new();
    learner.train_args.state_dim = 2;
    learner.train_args.train_rows = 16;
    learner.train_args.feature_columns = vec!["f1".to_string(), "f2".to_string()];
    learner.train_args.backend = "quadratic_q_cpu".to_string();
    learner.train_args.device_policy = "cpu".to_string();
    learner.train_args.requested_backend = Some("rlkit".to_string());
    learner.train_args.requested_device_policy = Some("cuda:0".to_string());
    learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
    learner.train_args.effective_device_policy = Some("cpu".to_string());
    learner.train_args.fallback_basis = TradingFallbackBasis::Quadratic;
    learner.train_args.training_report = Some(TradingRlTrainingReport {
        train_rows: 16,
        episode_count: 2,
        state_dim: 2,
        reward_horizon: 0,
        episode_len: 0,
        backend: "quadratic_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        average_hold_reward: 0.0,
        average_buy_reward: 0.0,
        average_sell_reward: 0.0,
        used_network_snapshot: false,
        used_fallback_q: true,
        used_feature_scaler: true,
    });
    learner.bounds = Some(FeatureBounds {
        mins: vec![0.0, 0.0],
        maxs: vec![1.0, 1.0],
    });
    learner.feature_columns = vec!["f1".to_string(), "f2".to_string()];
    learner.fallback_weights = Some(Array2::zeros((3, 4)));
    learner.fallback_bias = Some(Array1::zeros(3));
    learner.feature_scaler = Some(FeatureScaler {
        means: vec![0.25, 0.75],
        stds: vec![0.5, 0.25],
    });
    learner.training_report = Some(TradingRlTrainingReport {
        train_rows: 16,
        episode_count: 2,
        state_dim: 2,
        reward_horizon: 0,
        episode_len: 0,
        backend: "stale_backend".to_string(),
        device_policy: "stale_device".to_string(),
        average_hold_reward: 0.0,
        average_buy_reward: 0.0,
        average_sell_reward: 0.0,
        used_network_snapshot: true,
        used_fallback_q: false,
        used_feature_scaler: true,
    });

    let err = learner
        .artifact()
        .expect_err("live/persisted training report drift should be rejected");
    assert!(err.to_string().contains("training_report drifted"));
}

#[test]
fn artifact_rejects_missing_persisted_training_report() {
    let mut learner = TradingReinforcementLearner::new();
    learner.train_args.state_dim = 2;
    learner.train_args.train_rows = 16;
    learner.train_args.feature_columns = vec!["f1".to_string(), "f2".to_string()];
    learner.train_args.backend = "quadratic_q_cpu".to_string();
    learner.train_args.device_policy = "cpu".to_string();
    learner.train_args.requested_backend = Some("rlkit".to_string());
    learner.train_args.requested_device_policy = Some("cuda:0".to_string());
    learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
    learner.train_args.effective_device_policy = Some("cpu".to_string());
    learner.train_args.fallback_basis = TradingFallbackBasis::Quadratic;
    learner.train_args.training_report = None;
    learner.bounds = Some(FeatureBounds {
        mins: vec![0.0, 0.0],
        maxs: vec![1.0, 1.0],
    });
    learner.feature_columns = vec!["f1".to_string(), "f2".to_string()];
    learner.fallback_weights = Some(Array2::zeros((3, 4)));
    learner.fallback_bias = Some(Array1::zeros(3));
    learner.training_report = Some(TradingRlTrainingReport {
        train_rows: 16,
        episode_count: 2,
        state_dim: 2,
        reward_horizon: 0,
        episode_len: 0,
        backend: "quadratic_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        average_hold_reward: 0.0,
        average_buy_reward: 0.0,
        average_sell_reward: 0.0,
        used_network_snapshot: false,
        used_fallback_q: true,
        used_feature_scaler: false,
    });

    let err = learner
        .artifact()
        .expect_err("missing persisted training report should be rejected");
    assert!(
        err.to_string()
            .contains("missing the persisted training_report")
    );
}

#[test]
fn load_rejects_missing_network_snapshot_when_report_claims_one() {
    let path = unique_temp_dir("rl-missing-network-snapshot");
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 64,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("rlkit_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "rlkit_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 0,
        episode_len: 0,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "rlkit_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: true,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };
    let metadata = rl_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)
        .expect("build metadata");
    write_json(&path.join(METADATA_FILE_NAME), &metadata).expect("write metadata");
    write_json(&path.join("rl_config.json"), &artifact).expect("write config");

    let err = match TradingReinforcementLearner::load(&path) {
        Ok(_) => panic!("load should reject missing network snapshot"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("claims a network snapshot"));

    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn load_reconstructs_runtime_metadata_when_sidecar_missing() {
    let path = unique_temp_dir("rl-missing-metadata-sidecar");
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 64,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("rlkit_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "rlkit_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 0,
        episode_len: 0,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "rlkit_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };

    write_json(&path.join("rl_config.json"), &artifact).expect("write config");

    let loaded = TradingReinforcementLearner::load(&path)
        .expect("load should reconstruct metadata from artifact fields");
    assert_eq!(loaded.feature_columns, artifact.feature_columns);

    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn load_rejects_metadata_sidecar_drift_against_reconstructed_runtime_metadata() {
    let path = unique_temp_dir("rl-sidecar-drift");
    let artifact = TradingRlArtifact {
        state_dim: 2,
        feature_columns: vec!["f1".to_string(), "f2".to_string()],
        train_rows: 32,
        hidden_dims: vec![4, 4],
        state_encoding: TradingStateEncoding::Normalized,
        state_bins: 255,
        state_mins: vec![0.0, 0.0],
        state_maxs: vec![1.0, 1.0],
        buffer_capacity: 50_000,
        epochs: 64,
        max_steps: 512,
        update_interval: 32,
        update_freq: 4,
        batch_size: 64,
        learning_rate: 1e-3,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.02,
        epsilon_decay: 0.995,
        requested_backend: Some("rlkit".to_string()),
        requested_device_policy: Some("cpu".to_string()),
        effective_backend: Some("rlkit_cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        network_precision: None,
        backend: "rlkit_cpu".to_string(),
        device_policy: "cpu".to_string(),
        parallel_envs: 1,
        eval_episodes: 8,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: 1,
        reward_horizon: 0,
        episode_len: 0,
        training_report: Some(TradingRlTrainingReport {
            train_rows: 32,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "rlkit_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        }),
        feature_scaler: None,
        fallback_basis: TradingFallbackBasis::Quadratic,
        fallback_weights: Some(Array2::zeros((3, 4))),
        fallback_bias: Some(Array1::zeros(3)),
    };
    let mut drifted_metadata =
        rl_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)
            .expect("build metadata");
    drifted_metadata.training_summary.train_rows = 31;
    drifted_metadata.training_summary.val_rows = 1;
    drifted_metadata.training_summary.dataset_rows = 33;

    write_json(&path.join("rl_config.json"), &artifact).expect("write config");
    write_json(&path.join(METADATA_FILE_NAME), &drifted_metadata).expect("write metadata");

    let err = match TradingReinforcementLearner::load(&path) {
        Ok(_) => panic!("drifted sidecar should fail load"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("metadata sidecar mismatch"));

    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn preprocess_runtime_state_applies_persisted_feature_scaler() {
    let mut learner = TradingReinforcementLearner::new();
    learner.feature_scaler = Some(FeatureScaler {
        means: vec![1.0, 2.0],
        stds: vec![2.0, 4.0],
    });

    let scaled = learner
        .preprocess_runtime_state(&[5.0, 10.0])
        .expect("runtime preprocessing should use persisted scaler");
    assert_eq!(scaled, vec![2.0, 2.0]);
}

#[test]
fn predict_runtime_rejects_missing_bounds_before_inference() -> Result<()> {
    let mut learner = TradingReinforcementLearner::new();
    learner.train_args.state_dim = 1;
    learner.train_args.train_rows = 8;
    learner.train_args.feature_columns = vec!["f1".to_string()];
    learner.train_args.state_mins = vec![0.0];
    learner.train_args.state_maxs = vec![1.0];
    learner.train_args.backend = "linear_q_cpu".to_string();
    learner.train_args.device_policy = "cpu".to_string();
    learner.train_args.requested_backend = Some("linear_q_cpu".to_string());
    learner.train_args.requested_device_policy = Some("cpu".to_string());
    learner.train_args.effective_backend = Some("linear_q_cpu".to_string());
    learner.train_args.effective_device_policy = Some("cpu".to_string());
    learner.train_args.training_report = Some(TradingRlTrainingReport {
        train_rows: 8,
        episode_count: 1,
        state_dim: 1,
        reward_horizon: learner.train_args.reward_horizon,
        episode_len: learner.train_args.episode_len,
        backend: "linear_q_cpu".to_string(),
        device_policy: "cpu".to_string(),
        average_hold_reward: 0.0,
        average_buy_reward: 0.0,
        average_sell_reward: 0.0,
        used_network_snapshot: false,
        used_fallback_q: true,
        used_feature_scaler: false,
    });
    learner.feature_columns = learner.train_args.feature_columns.clone();
    learner.fallback_weights = Some(Array2::zeros((3, 1)));
    learner.fallback_bias = Some(Array1::zeros(3));
    learner.training_report = learner.train_args.training_report.clone();
    let df = DataFrame::new(vec![Series::new("f1".into(), vec![0.0_f64]).into()])?;

    let err = learner
        .predict_runtime(&df)
        .expect_err("missing bounds should fail early");
    assert!(err.to_string().contains("feature bounds"));
    Ok(())
}
