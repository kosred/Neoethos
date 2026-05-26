#![allow(clippy::field_reassign_with_default)]

// TODO(real-data): every Experience / state vector / reward in this
// file is synthetic (e.g. `vec![-10.0, 1.0, 2.0, 3.0, 4.0, 5.0]`).
// Replace with a cTrader-sourced exit-decision sample: real
// trade-state vectors recorded from a backtest over the target
// symbol/timeframe so the regret / propagation paths fire on
// realistic outcomes.
// `use super::*;` removed 2026-05-26 — nothing actually consumed from it; the
// explicit import below covers every symbol referenced in the test bodies.
use super::{ExitAgent, ExitAgentArtifact, Experience, PendingRegret, exit_runtime_metadata};
use crate::base::three_class_runtime_confidence;
use crate::statistical::common::{METADATA_FILE_NAME, write_json};
use anyhow::Result;
use burn::module::Param;
use burn::tensor::Tensor;
use polars::prelude::{DataFrame, NamedFrom, Series};
use std::path::PathBuf;

#[test]
fn observe_exit_uses_explicit_direction() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    let state = vec![-10.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    agent.observe_exit(7, &state, 0, 1, 1.2345, 42);

    let pending = agent
        .pending_regret
        .get(&7)
        .expect("pending regret should be stored");
    assert_eq!(pending.direction, 1);
}

#[test]
fn process_regret_keeps_pending_when_future_trace_is_empty() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    let state = vec![1.0, 0.2, 0.3, 0.4, 0.5, 0.6];

    agent.observe_exit(11, &state, 1, -1, 1.2000, 100);
    agent.process_regret(11, &[]);

    assert!(
        agent.pending_regret.contains_key(&11),
        "empty future trace should not consume the pending regret"
    );
}

#[test]
fn reward_from_trace_prefers_hold_when_favorable_move_dominates() {
    let reward = ExitAgent::reward_from_trace(1, 1.2000, &[1.2050, 1.2040, 1.1980], 0);
    assert!(
        reward > 0.0,
        "hold should be rewarded when upside dominates"
    );

    let close_reward = ExitAgent::reward_from_trace(1, 1.2000, &[1.2050, 1.2040, 1.1980], 1);
    assert!(
        close_reward < 0.0,
        "closing should be penalized when upside dominates"
    );
}

#[test]
fn runtime_probabilities_keep_exit_agent_mapping_truthful() {
    let mapped = ExitAgent::runtime_probabilities(0.7, 0.2);
    assert!(mapped[0] > mapped[2]);
    assert!(mapped[1] > 0.0);
    let total = mapped.iter().sum::<f32>();
    assert!((total - 1.0).abs() < 1e-6);
}

#[test]
fn runtime_probabilities_assign_more_neutral_mass_when_indecisive() {
    let decisive = ExitAgent::runtime_probabilities(0.8, 0.2);
    let indecisive = ExitAgent::runtime_probabilities(0.51, 0.49);
    assert!(
        indecisive[1] > decisive[1],
        "neutral mass should grow when hold and close are nearly tied"
    );
}

#[test]
fn validated_runtime_probabilities_rejects_degenerate_rows() {
    let err = ExitAgent::validated_runtime_probabilities(&[0.0, 0.0])
        .expect_err("zero-mass probabilities should be rejected");
    assert!(
        err.to_string().contains("degenerate probability mass"),
        "unexpected error: {err}"
    );
}

#[test]
fn load_rejects_replay_memory_longer_than_recorded_training_rows() {
    let path = unique_temp_dir("exit-agent-train-rows");
    let artifact = ExitAgentArtifact {
        input_dim: 6,
        hidden_dim: 16,
        feature_columns: vec![
            "f1".to_string(),
            "f2".to_string(),
            "f3".to_string(),
            "f4".to_string(),
            "f5".to_string(),
            "f6".to_string(),
        ],
        gamma: 0.99,
        epsilon: 0.2,
        epsilon_min: 0.05,
        epsilon_decay: 0.999,
        memory_capacity: 1_024,
        reward_horizon: 0,
        warmup_steps: 0,
        train_rows: 1,
        trained_memory_size: 2,
        average_reward: 0.0,
        training_report: Some(super::ExitAgentTrainingReport {
            train_rows: 1,
            memory_size: 2,
            warmup_steps: 0,
            average_reward: 0.0,
            reward_horizon: 0,
            feature_count: 6,
            requested_device_policy: "cpu".to_string(),
            effective_device_policy: "cpu".to_string(),
            execution_backend: "burn_ndarray".to_string(),
        }),
        requested_device_policy: Some("cpu".to_string()),
        effective_device_policy: Some("cpu".to_string()),
        execution_backend: Some("burn_ndarray".to_string()),
        runtime_metadata: None,
        replay_memory: vec![
            Experience {
                state: vec![0.0; 6],
                next_state: None,
                action: 0,
                reward: 0.0,
                done: true,
            },
            Experience {
                state: vec![0.0; 6],
                next_state: None,
                action: 1,
                reward: 1.0,
                done: true,
            },
        ],
        pending_regret: Default::default(),
    };
    std::fs::write(
        ExitAgent::artifact_path(&path),
        serde_json::to_vec_pretty(&artifact).expect("serialize artifact"),
    )
    .expect("write config");
    write_json(
        &path.join(METADATA_FILE_NAME),
        &exit_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)
            .expect("build metadata"),
    )
    .expect("write metadata");

    let err = ExitAgent::load(&path)
        .err()
        .expect("inconsistent train rows should fail");
    assert!(
        err.to_string().contains("train_rows"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn load_uses_embedded_runtime_metadata_when_sidecar_missing() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16).with_memory_capacity(1_024);
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.train_rows = 64;
    agent.trained_checkpoint_ready = true;
    agent.push_experience(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 1,
        reward: 0.5,
        done: true,
    });
    attach_training_report(&mut agent, 0.5);
    let path = unique_temp_dir("exit-agent-embedded-metadata");
    agent.save(&path).expect("save should succeed");
    std::fs::remove_file(path.join(METADATA_FILE_NAME)).expect("remove metadata sidecar");

    let loaded = ExitAgent::load(&path).expect("load should fallback to embedded metadata");
    assert_eq!(loaded.train_rows, 64);
    assert_eq!(loaded.feature_columns.len(), 6);

    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn load_rejects_sidecar_embedded_runtime_metadata_drift() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16).with_memory_capacity(1_024);
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.train_rows = 64;
    agent.trained_checkpoint_ready = true;
    agent.push_experience(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 1,
        reward: 0.5,
        done: true,
    });
    attach_training_report(&mut agent, 0.5);
    let path = unique_temp_dir("exit-agent-metadata-drift");
    agent.save(&path).expect("save should succeed");

    let mut drifted = exit_runtime_metadata(agent.feature_columns.clone(), agent.train_rows)
        .expect("build metadata");
    drifted.training_summary.train_rows = drifted.training_summary.train_rows.saturating_sub(1);
    drifted.training_summary.val_rows = drifted.training_summary.val_rows.saturating_add(1);
    write_json(&path.join(METADATA_FILE_NAME), &drifted).expect("overwrite drifted sidecar");

    let err = ExitAgent::load(&path)
        .err()
        .expect("drifted sidecar metadata should be rejected");
    assert!(err.to_string().contains("drift"), "unexpected error: {err}");

    let _ = std::fs::remove_dir_all(&path);
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{stamp}-{}", std::process::id()));
    std::fs::create_dir_all(&path).expect("create temp directory");
    path
}

fn attach_training_report(agent: &mut ExitAgent, average_reward: f32) {
    agent.trained_memory_size = agent.memory.len();
    agent.average_reward = average_reward;
    agent.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: agent.train_rows,
        memory_size: agent.memory.len(),
        warmup_steps: agent.warmup_steps,
        average_reward,
        reward_horizon: agent.reward_horizon,
        feature_count: agent.input_dim,
        requested_device_policy: agent.requested_device_policy.clone(),
        effective_device_policy: agent.effective_device_policy.clone(),
        execution_backend: agent.execution_backend.clone(),
    });
}

#[test]
fn process_regret_respects_configured_memory_capacity() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16).with_memory_capacity(1_024);
    let state = vec![1.0, 0.2, 0.3, 0.4, 0.5, 0.6];

    for ticket in 0..1_025 {
        agent.observe_exit(
            ticket,
            &state,
            0,
            1,
            1.2 + ticket as f64 * 0.001,
            ticket as i64,
        );
        agent.process_regret(ticket, &[1.2050, 1.2040, 1.1980]);
    }

    assert_eq!(agent.memory_size(), 1_024);
}

#[test]
fn save_and_load_preserve_memory_capacity() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16).with_memory_capacity(1_024);
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.train_rows = 64;
    agent.push_experience(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 0,
        reward: 0.25,
        done: true,
    });
    agent.trained_memory_size = agent.memory.len();
    agent.average_reward = 0.25;
    agent.trained_checkpoint_ready = true;
    agent.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 64,
        memory_size: agent.memory.len(),
        warmup_steps: 0,
        average_reward: 0.25,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "auto".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: agent.execution_backend.clone(),
    });
    let path = unique_temp_dir("exit-agent-capacity");

    agent.save(&path).expect("save should succeed");
    let loaded = ExitAgent::load(&path).expect("load should succeed");

    assert_eq!(loaded.artifact().memory_capacity, 1_024);
    assert!(path.join("optimizer.mpk").exists());
    assert!(
        loaded.memory.capacity() >= 1_024,
        "loaded memory should honor the configured minimum capacity"
    );
    assert_eq!(
        loaded
            .training_report
            .as_ref()
            .expect("training report should round-trip")
            .train_rows,
        64
    );

    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn save_rejects_synthetic_trained_state_without_replay_memory() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.train_rows = 64;
    agent.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 64,
        memory_size: 0,
        warmup_steps: 0,
        average_reward: 0.0,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "auto".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: agent.execution_backend.clone(),
    });
    let path = unique_temp_dir("exit-agent-synthetic-trained");

    let err = agent
        .save(&path)
        .expect_err("synthetic trained state without replay memory should fail");
    assert!(
        err.to_string().contains("untrained runtime state")
            || err.to_string().contains("zero training-report memory_size")
            || err.to_string().contains("empty replay memory"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn predict_runtime_uses_shared_three_class_confidence_gate() -> Result<()> {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    let device = agent.device;
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];

    agent.model.fc1.weight = Param::from_tensor(Tensor::from_data([[0.0_f32; 16]; 6], &device));
    agent.model.fc1.bias = Some(Param::from_tensor(Tensor::from_data(
        [0.0_f32; 16],
        &device,
    )));
    agent.model.fc2.weight = Param::from_tensor(Tensor::from_data([[0.0_f32; 16]; 16], &device));
    agent.model.fc2.bias = Some(Param::from_tensor(Tensor::from_data(
        [0.0_f32; 16],
        &device,
    )));
    agent.model.output.weight = Param::from_tensor(Tensor::from_data([[0.0_f32; 2]; 16], &device));
    agent.model.output.bias = Some(Param::from_tensor(Tensor::from_data(
        [0.84729785_f32, 0.0],
        &device,
    )));
    agent.train_rows = 64;
    agent.trained_memory_size = 1;
    agent.push_experience(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 0,
        reward: 0.1,
        done: true,
    });
    agent.trained_checkpoint_ready = true;
    agent.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 64,
        memory_size: 1,
        warmup_steps: 0,
        average_reward: 0.1,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "auto".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: agent.execution_backend.clone(),
    });

    let df = DataFrame::new(vec![
        Series::new("f1".into(), vec![0.0_f64]).into(),
        Series::new("f2".into(), vec![0.0_f64]).into(),
        Series::new("f3".into(), vec![0.0_f64]).into(),
        Series::new("f4".into(), vec![0.0_f64]).into(),
        Series::new("f5".into(), vec![0.0_f64]).into(),
        Series::new("f6".into(), vec![0.0_f64]).into(),
    ])?;

    let predictions = agent.predict_runtime(&df)?;
    let prediction = predictions
        .first()
        .expect("one runtime prediction should be produced");
    let expected_row = ExitAgent::runtime_probabilities(0.7, 0.3);
    let (expected_confidence, expected_abstain) = three_class_runtime_confidence(expected_row)?;

    assert!((prediction.confidence().expect("confidence") - expected_confidence).abs() < 1e-6);
    assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
    assert!(prediction.metadata().execution_backend.is_some());
    Ok(())
}

#[test]
fn artifact_carries_device_policy_fields() {
    let agent = ExitAgent::with_hidden_dim(6, 16).with_device_policy("cpu");
    let artifact = agent.artifact();
    assert_eq!(artifact.requested_device_policy.as_deref(), Some("cpu"));
    assert_eq!(artifact.effective_device_policy.as_deref(), Some("cpu"));
    assert!(artifact.execution_backend.is_some());
}

#[test]
fn with_device_policy_invalidates_trained_runtime_state() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.train_rows = 64;
    agent.trained_memory_size = 1;
    agent.average_reward = 0.25;
    agent.push_experience(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 0,
        reward: 0.25,
        done: true,
    });
    agent.pending_regret.insert(
        7,
        PendingRegret {
            state: vec![0.0; 6],
            action: 1,
            exit_price: 1.2,
            time: 42,
            direction: 1,
        },
    );
    agent.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 64,
        memory_size: 1,
        warmup_steps: 0,
        average_reward: 0.25,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "auto".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: agent.execution_backend.clone(),
    });
    agent.trained_checkpoint_ready = true;
    agent.persisted_requested_device_policy = Some("auto".to_string());
    agent.persisted_effective_device_policy = Some("cpu".to_string());
    agent.persisted_execution_backend = Some(agent.execution_backend.clone());

    let agent = agent.with_device_policy("cpu");

    assert_eq!(agent.train_rows, 0);
    assert_eq!(agent.trained_memory_size, 0);
    assert_eq!(agent.average_reward, 0.0);
    assert!(agent.memory.is_empty());
    assert!(agent.pending_regret.is_empty());
    assert!(agent.training_report.is_none());
    assert!(!agent.trained_checkpoint_ready);
    assert!(agent.persisted_requested_device_policy.is_none());
    assert!(agent.persisted_effective_device_policy.is_none());
    assert!(agent.persisted_execution_backend.is_none());
}

#[test]
fn set_epsilon_clamps_and_ignores_non_finite_values() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16).with_exploration_schedule(0.05, 0.999);
    agent.set_epsilon(2.0);
    assert_eq!(agent.get_epsilon(), 1.0);

    agent.set_epsilon(-1.0);
    assert_eq!(agent.get_epsilon(), 0.05);

    agent.set_epsilon(f32::NAN);
    assert_eq!(agent.get_epsilon(), 0.05);
}

#[test]
fn validate_exit_artifact_rejects_invalid_pending_regret_direction() {
    let mut artifact = ExitAgentArtifact::default();
    artifact.input_dim = 6;
    artifact.hidden_dim = 16;
    artifact.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    artifact.train_rows = 64;
    artifact.pending_regret.insert(
        7,
        super::PendingRegret {
            state: vec![0.0; 6],
            action: 0,
            exit_price: 1.2,
            time: 42,
            direction: 0,
        },
    );

    let err = super::validate_exit_artifact(&artifact)
        .expect_err("invalid pending regret direction should fail");
    assert!(err.to_string().contains("direction"));
}

#[test]
fn validate_exit_artifact_rejects_partial_runtime_identity_triplet() {
    let mut artifact = ExitAgentArtifact::default();
    artifact.input_dim = 6;
    artifact.hidden_dim = 16;
    artifact.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    artifact.train_rows = 64;
    artifact.trained_memory_size = 1;
    artifact.replay_memory.push(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 0,
        reward: 0.0,
        done: true,
    });
    artifact.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 64,
        memory_size: 1,
        warmup_steps: 0,
        average_reward: 0.0,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "cpu".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: "burn_ndarray".to_string(),
    });
    artifact.requested_device_policy = Some("cpu".to_string());
    artifact.effective_device_policy = None;
    artifact.execution_backend = Some("burn_ndarray".to_string());

    let err = super::validate_exit_artifact(&artifact)
        .expect_err("partial runtime identity triplet should fail");
    assert!(err.to_string().contains("requested_device_policy"));
}

#[test]
fn validate_exit_artifact_rejects_training_report_warmup_mismatch() {
    let mut artifact = ExitAgentArtifact::default();
    artifact.input_dim = 6;
    artifact.hidden_dim = 16;
    artifact.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    artifact.train_rows = 64;
    artifact.trained_memory_size = 1;
    artifact.warmup_steps = 32;
    artifact.replay_memory.push(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 0,
        reward: 0.0,
        done: true,
    });
    artifact.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 64,
        memory_size: 1,
        warmup_steps: 16,
        average_reward: 0.0,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "cpu".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: "burn_ndarray".to_string(),
    });

    let err = super::validate_exit_artifact(&artifact).expect_err("warmup mismatch should fail");
    assert!(err.to_string().contains("warmup_steps"));
}

#[test]
fn validate_exit_artifact_rejects_average_reward_drift() {
    let mut artifact = ExitAgentArtifact::default();
    artifact.input_dim = 6;
    artifact.hidden_dim = 16;
    artifact.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    artifact.train_rows = 64;
    artifact.trained_memory_size = 1;
    artifact.average_reward = 0.25;
    artifact.replay_memory.push(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 0,
        reward: 0.25,
        done: true,
    });
    artifact.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 64,
        memory_size: 1,
        warmup_steps: 0,
        average_reward: 0.1,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "cpu".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: "burn_ndarray".to_string(),
    });
    artifact.requested_device_policy = Some("cpu".to_string());
    artifact.effective_device_policy = Some("cpu".to_string());
    artifact.execution_backend = Some("burn_ndarray".to_string());

    let err =
        super::validate_exit_artifact(&artifact).expect_err("average_reward drift should fail");
    assert!(err.to_string().contains("average_reward"));
}

#[test]
fn validate_exit_artifact_rejects_replay_memory_size_drift() {
    let mut artifact = ExitAgentArtifact::default();
    artifact.input_dim = 6;
    artifact.hidden_dim = 16;
    artifact.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    artifact.train_rows = 64;
    artifact.trained_memory_size = 2;
    artifact.average_reward = 0.0;
    artifact.replay_memory.push(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 0,
        reward: 0.0,
        done: true,
    });
    artifact.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 64,
        memory_size: 2,
        warmup_steps: 0,
        average_reward: 0.0,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "cpu".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: "burn_ndarray".to_string(),
    });
    artifact.requested_device_policy = Some("cpu".to_string());
    artifact.effective_device_policy = Some("cpu".to_string());
    artifact.execution_backend = Some("burn_ndarray".to_string());

    let err =
        super::validate_exit_artifact(&artifact).expect_err("replay memory size drift should fail");
    assert!(err.to_string().contains("trained_memory_size"));
}

#[test]
fn runtime_degraded_reason_reports_missing_training_report_for_trained_state() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    agent.train_rows = 128;
    agent.trained_memory_size = 64;
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.training_report = None;

    let degraded_reason = agent
        .runtime_degraded_reason()
        .expect("trained agent without report should be marked degraded");
    assert!(degraded_reason.contains("persisted training report"));
}

#[test]
fn runtime_degraded_reason_reports_zero_persisted_replay_memory() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    agent.train_rows = 128;
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 128,
        memory_size: 0,
        warmup_steps: 0,
        average_reward: 0.0,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "auto".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: agent.execution_backend.clone(),
    });
    agent.trained_memory_size = 0;

    let degraded_reason = agent
        .runtime_degraded_reason()
        .expect("zero-memory trained state should be degraded");
    assert!(degraded_reason.contains("zero persisted replay memory"));
}

#[test]
fn runtime_degraded_reason_reports_missing_trained_checkpoint() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    agent.train_rows = 128;
    agent.trained_memory_size = 1;
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.push_experience(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 0,
        reward: 0.0,
        done: true,
    });
    agent.training_report = Some(super::ExitAgentTrainingReport {
        train_rows: 128,
        memory_size: 1,
        warmup_steps: 0,
        average_reward: 0.0,
        reward_horizon: 0,
        feature_count: 6,
        requested_device_policy: "auto".to_string(),
        effective_device_policy: "cpu".to_string(),
        execution_backend: agent.execution_backend.clone(),
    });

    let degraded_reason = agent
        .runtime_degraded_reason()
        .expect("missing trained checkpoint should be degraded");
    assert!(degraded_reason.contains("verified trained checkpoint"));
}

#[test]
fn runtime_degraded_reason_marks_untrained_runtime_state() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.train_rows = 64;

    let degraded_reason = agent
        .runtime_degraded_reason()
        .expect("cold trained metadata should be marked degraded");
    assert!(degraded_reason.contains("not trained enough for inference"));
}

#[test]
fn predict_runtime_rejects_untrained_runtime_state() {
    let agent = ExitAgent::with_hidden_dim(6, 16);
    let df = DataFrame::new(vec![
        Series::new("f1".into(), vec![0.0_f64]).into(),
        Series::new("f2".into(), vec![0.1_f64]).into(),
        Series::new("f3".into(), vec![0.2_f64]).into(),
        Series::new("f4".into(), vec![0.3_f64]).into(),
        Series::new("f5".into(), vec![0.4_f64]).into(),
        Series::new("f6".into(), vec![0.5_f64]).into(),
    ])
    .expect("single-row frame");

    let err = agent
        .predict_runtime(&df)
        .expect_err("cold exit agent should not run inference");
    assert!(err.to_string().contains("untrained runtime state"));
}

#[test]
fn predict_runtime_rejects_missing_training_report_in_trained_state() {
    let mut agent = ExitAgent::with_hidden_dim(6, 16);
    agent.feature_columns = vec![
        "f1".to_string(),
        "f2".to_string(),
        "f3".to_string(),
        "f4".to_string(),
        "f5".to_string(),
        "f6".to_string(),
    ];
    agent.train_rows = 64;
    agent.trained_memory_size = 1;
    agent.push_experience(Experience {
        state: vec![0.0; 6],
        next_state: None,
        action: 0,
        reward: 0.1,
        done: true,
    });
    agent.trained_checkpoint_ready = true;
    agent.training_report = None;

    let df = DataFrame::new(vec![
        Series::new("f1".into(), vec![0.0_f64]).into(),
        Series::new("f2".into(), vec![0.1_f64]).into(),
        Series::new("f3".into(), vec![0.2_f64]).into(),
        Series::new("f4".into(), vec![0.3_f64]).into(),
        Series::new("f5".into(), vec![0.4_f64]).into(),
        Series::new("f6".into(), vec![0.5_f64]).into(),
    ])
    .expect("single-row frame");

    let err = agent
        .predict_runtime(&df)
        .expect_err("trained state without training report must fail");
    assert!(err.to_string().contains("persisted training report"));
}
