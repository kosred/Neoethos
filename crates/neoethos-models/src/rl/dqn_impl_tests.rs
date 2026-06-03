// TODO(real-data): every DataFrame, weight matrix, Q-value vector and
// feature normalisation range in this file is synthetic
// (`vec![0.0, 0.0]`, `vec![1.0, 1.0]`, hand-written Q-value targets
// like 0.35 / 0.95 / 0.4). Replace each fixture below with a cTrader
// historical sample for the symbol/timeframe the DQN learner is
// targeted at (e.g. EURUSD M15 features built from real OHLCV), so
// asserted Q-values come from real state distributions rather than
// algebraic identities.
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
fn fallback_only_artifact_serde_roundtrip() {
    // F-MODELS9-004 honesty note: this test ONLY asserts that the
    // fallback-only learner artifact serialises, reloads, and the
    // fallback inference path produces the algebraically-expected
    // output for the hardcoded weights/bias scaffold. It does NOT
    // validate that the per-action average rewards stored in the
    // training report are correct for any real training dataset:
    // the `average_buy_reward = 0.2 > average_hold_reward = 0.1 >
    // average_sell_reward = -0.1` triplet is pre-baked into the
    // metadata before `save()` is called, so the test would still
    // pass if the production reward function were entirely broken.
    // The aspirational learning test that exercises
    // `build_reward_triplet` end-to-end on a real cTrader fixture
    // is registered below as `#[ignore]` and tracks F-MODELS9-004
    // until the historical-data fixture is available.
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
    learner.fallback_weights = Some(weights.clone());
    learner.fallback_bias = Some(bias.clone());
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

    // F-MODELS9-001 fix: derive expected values inline from weights + bias
    // instead of asserting hardcoded magic numbers. The forward pass is
    // `q = W @ x + b` where W is row-major [3, 2], x is [2], b is [3].
    // If someone refactors the fallback inference path, this test will
    // catch the change because the formula is checked, not the result.
    let input = [0.25_f32, 0.75_f32];
    let expected_q = [
        weights[(0, 0)] * input[0] + weights[(0, 1)] * input[1] + bias[0],
        weights[(1, 0)] * input[0] + weights[(1, 1)] * input[1] + bias[1],
        weights[(2, 0)] * input[0] + weights[(2, 1)] * input[1] + bias[2],
    ];
    assert_eq!(q_values.len(), 3);
    for (i, expected) in expected_q.iter().enumerate() {
        assert!(
            (q_values[i] - expected).abs() < 1e-6,
            "q_values[{i}] = {} but expected {} = W[{i}] @ {input:?} + b[{i}]",
            q_values[i],
            expected
        );
    }
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

#[test]
#[ignore = "needs real cTrader-historical fixture and a full training pass; tracks F-MODELS9-004"]
fn trained_agent_learns_action_reward_ordering_on_real_data() {
    // F-MODELS9-004 aspirational test.
    //
    // Goal: drive a full DQN training pass over a real cTrader
    // historical sample (e.g. EURUSD M15 OHLCV) so that the
    // production reward computation in `build_reward_triplet` is
    // exercised end-to-end, then assert that the resulting
    // `TradingRlTrainingReport` exhibits the expected
    // action-reward ordering for the chosen regime
    // (`average_buy_reward > average_hold_reward > average_sell_reward`
    // for a bullish window, with the inequality flipped for a
    // bearish window).
    //
    // This test is intentionally `#[ignore]` because the operator
    // directive forbids synthetic broker data; it cannot be enabled
    // until a real cTrader historical fixture is checked into the
    // workspace and a deterministic training-pass harness is wired
    // up to consume it. Once those are available, replace the
    // unimplemented! below with a fixture loader + a learner
    // `fit_*` invocation and the corresponding ordering assertions.
    unimplemented!(
        "F-MODELS9-004: blocked on real cTrader-historical fixture + \
         deterministic full-training harness; do not enable with \
         synthetic data"
    );
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
fn normalize_rl_device_policy_edge_cases() {
    // Case sensitivity: lowercase, uppercase, and mixed-case spellings of
    // the same vendor+index pair must collapse to the same canonical form.
    // The normaliser calls `trim().to_ascii_lowercase()` first, so casing
    // is irrelevant to the result.
    assert_eq!(normalize_rl_device_policy("cuda:0"), "gpu:0");
    assert_eq!(normalize_rl_device_policy("CUDA:0"), "gpu:0");
    assert_eq!(normalize_rl_device_policy("CuDa:0"), "gpu:0");

    // Bare lower-case vendor name (no colon, no index) — vendor is in
    // the recognised set, so it collapses to the canonical "gpu" token
    // meaning "pick a default GPU device".
    assert_eq!(normalize_rl_device_policy("rocm"), "gpu");

    // CPU pass-through: the literal "cpu" token is canonical and must
    // survive all of: lowercase, uppercase, surrounding whitespace.
    assert_eq!(normalize_rl_device_policy("cpu"), "cpu");
    assert_eq!(normalize_rl_device_policy("CPU"), "cpu");
    assert_eq!(normalize_rl_device_policy("  cpu  "), "cpu");

    // Leading/trailing whitespace is stripped by `trim()` before the
    // prefix match, so the canonical form is the same as the trimmed
    // input.
    assert_eq!(normalize_rl_device_policy("  cuda:0  "), "gpu:0");

    // Contract: an internal space inside the vendor token ("cuda :0")
    // is NOT a recognised form — `trim()` only removes leading/trailing
    // whitespace, the literal "cuda :0" does not match any of the
    // vendor prefixes, and the RL whitelist then rewrites it to "auto"
    // (strict whitelist semantics specific to the RL normaliser).
    assert_eq!(normalize_rl_device_policy("cuda :0"), "auto");

    // Empty input defaults to "auto" — let the runtime pick a device.
    assert_eq!(normalize_rl_device_policy(""), "auto");
    // Whitespace-only input is treated the same as empty after `trim()`.
    assert_eq!(normalize_rl_device_policy("   "), "auto");

    // Observation (NOT a bug): the normaliser performs NO bounds-check
    // on the index suffix. "cuda:99" yields "gpu:99" even if the host
    // has fewer GPUs, and "cuda:" (empty index) yields the syntactically
    // odd "gpu:". The downstream GPU dispatch layer is responsible for
    // rejecting indices that don't map to a physical device; this
    // normaliser only handles vendor-alias collapsing.
    assert_eq!(normalize_rl_device_policy("cuda:99"), "gpu:99");
    assert_eq!(normalize_rl_device_policy("cuda:"), "gpu:");

    // RL-specific extension: `wgpu` (with or without index) is in the
    // extra-prefix list passed to the shared helper, so it normalises
    // to the canonical "gpu" / "gpu:N" form.
    assert_eq!(normalize_rl_device_policy("wgpu"), "gpu");
    assert_eq!(normalize_rl_device_policy("wgpu:2"), "gpu:2");

    // Unknown vendor tokens are forced to "auto" by the RL whitelist
    // (the shared helper would return them lowercased-but-unchanged,
    // but the RL wrapper layers strict validation on top).
    assert_eq!(normalize_rl_device_policy("tpu"), "auto");
    assert_eq!(normalize_rl_device_policy("nonsense"), "auto");
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
    // F-MODELS9-002 fix: previously only NaN was tested. Inf/NegInf
    // and mixed-finite-with-non-finite also reach this validator from
    // upstream training paths; cover the full boundary.
    let err =
        validate_q_values(vec![0.1, f32::NAN, 0.2]).expect_err("NaN q-values should be rejected");
    assert!(err.to_string().contains("non-finite"), "unexpected: {err}");

    let err = validate_q_values(vec![f32::INFINITY, 0.0, 0.0])
        .expect_err("+Inf q-values should be rejected");
    assert!(err.to_string().contains("non-finite"), "unexpected: {err}");

    let err = validate_q_values(vec![0.0, f32::NEG_INFINITY, 0.0])
        .expect_err("-Inf q-values should be rejected");
    assert!(err.to_string().contains("non-finite"), "unexpected: {err}");

    let err = validate_q_values(vec![1.0, 2.0, f32::INFINITY])
        .expect_err("mixed finite+Inf q-values should be rejected");
    assert!(err.to_string().contains("non-finite"), "unexpected: {err}");

    // All-zeros and all-finite must PASS (the validator should not be
    // over-eager and reject legitimate zero-policy actions).
    validate_q_values(vec![0.0, 0.0, 0.0]).expect("all-zeros must be accepted");
    validate_q_values(vec![0.3, -1.5, 2.7]).expect("finite values must be accepted");
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
        std::env::set_var("NEOETHOS_BOT_DQN_TRAIN_PRECISION", "bf16");
    }
    let learner = TradingReinforcementLearner::new();
    let (_backend, degraded_reason) = learner.runtime_backend_details();
    unsafe {
        std::env::remove_var("NEOETHOS_BOT_DQN_TRAIN_PRECISION");
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
    let (effective_precision, degraded_reason) =
        resolve_rl_training_precision_with_capability(Some("bf16"), "rlkit_cpu", "cpu", Some(true));

    assert_eq!(effective_precision, "bf16");
    assert!(degraded_reason.is_none());
}

#[test]
fn rl_precision_resolution_explains_cpu_backend_limit() {
    let (effective_precision, degraded_reason) =
        resolve_rl_training_precision_with_capability(Some("bf16"), "quadratic_q_cpu", "cpu", None);

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
    // F-MODELS9-005 fix: derive expected values inline from the z-score
    // formula `z = (x - mean) / std` instead of asserting hardcoded
    // numbers. If the scaler implementation drifts to min-max or any
    // other transform, this test will catch it because the formula
    // is checked, not the result.
    let mut learner = TradingReinforcementLearner::new();
    let means = vec![1.0_f32, 2.0];
    let stds = vec![2.0_f32, 4.0];
    learner.feature_scaler = Some(FeatureScaler {
        means: means.clone(),
        stds: stds.clone(),
    });

    let inputs = [5.0_f32, 10.0_f32];
    let scaled = learner
        .preprocess_runtime_state(&inputs)
        .expect("runtime preprocessing should use persisted scaler");
    let expected: Vec<f32> = inputs
        .iter()
        .zip(means.iter().zip(stds.iter()))
        .map(|(x, (mean, std))| (*x - *mean) / *std)
        .collect();
    assert_eq!(scaled.len(), expected.len());
    for (i, (s, e)) in scaled.iter().zip(expected.iter()).enumerate() {
        assert!(
            (s - e).abs() < 1e-6,
            "scaled[{i}] = {} but expected z-score = (x={} - mean={}) / std={} = {}",
            s,
            inputs[i],
            means[i],
            stds[i],
            e
        );
    }

    // Second case with non-trivial mean/std to prove the formula isn't
    // numerically coincidental with the first case (both inputs were
    // chosen to land on z=2.0 above; here z varies per feature).
    let means2 = vec![0.5_f32, -1.0];
    let stds2 = vec![0.25_f32, 3.0];
    learner.feature_scaler = Some(FeatureScaler {
        means: means2.clone(),
        stds: stds2.clone(),
    });
    let inputs2 = [1.0_f32, 5.0_f32];
    let scaled2 = learner
        .preprocess_runtime_state(&inputs2)
        .expect("second case must also work");
    let expected2: Vec<f32> = inputs2
        .iter()
        .zip(means2.iter().zip(stds2.iter()))
        .map(|(x, (mean, std))| (*x - *mean) / *std)
        .collect();
    for (i, (s, e)) in scaled2.iter().zip(expected2.iter()).enumerate() {
        assert!((s - e).abs() < 1e-6, "case 2, feature {i}");
    }
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
