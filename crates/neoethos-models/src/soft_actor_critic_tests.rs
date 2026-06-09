// SAC-Discrete correctness tests.
//
// TODO(real-data): the feature frames below are deterministic
// in-memory fixtures (sinusoidal price proxies + a triple-barrier-style
// label). Replace with a cTrader-sourced feature/label sample over the
// target symbol/timeframe so the episode/reward construction fires on
// realistic outcomes. These fixtures are training-mechanics tests only,
// not statistical-quality tests.

use super::{
    NUM_ACTIONS, SacNetConfig, SacTuple, SoftActorCritic, average_rewards, default_target_entropy,
    tuples_from_episodes,
};
use crate::burn_models::InferBackend;
use crate::rl::TradingTransition;
use anyhow::Result;
use burn::prelude::*;
use polars::prelude::{DataFrame, NamedFrom, Series};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{stamp}-{}", std::process::id()));
    std::fs::create_dir_all(&path).expect("create temp directory");
    path
}

/// Build a small deterministic feature frame + 3-class label series.
/// 4 features, `rows` rows, labels in {-1, 0, 1} on a slow cycle so the
/// reward triplets are non-degenerate.
fn fixture_frame(rows: usize) -> (DataFrame, Series) {
    let mut f0 = Vec::with_capacity(rows);
    let mut f1 = Vec::with_capacity(rows);
    let mut f2 = Vec::with_capacity(rows);
    let mut f3 = Vec::with_capacity(rows);
    let mut labels = Vec::with_capacity(rows);
    for i in 0..rows {
        let t = i as f64;
        f0.push((t * 0.10).sin());
        f1.push((t * 0.05).cos());
        f2.push((t % 7.0) / 7.0);
        f3.push(((t * 0.02).sin() * 0.5) + 0.5);
        // slow regime cycle: 24 up, 24 flat, 24 down, repeat
        let phase = (i / 24) % 3;
        labels.push(match phase {
            0 => 1_i32,
            1 => 0_i32,
            _ => -1_i32,
        });
    }
    let df = DataFrame::new(vec![
        Series::new("f0".into(), f0).into(),
        Series::new("f1".into(), f1).into(),
        Series::new("f2".into(), f2).into(),
        Series::new("f3".into(), f3).into(),
    ])
    .expect("build fixture frame");
    let labels = Series::new("label".into(), labels);
    (df, labels)
}

fn trained_agent(rows: usize) -> SoftActorCritic {
    let (df, labels) = fixture_frame(rows);
    let mut agent = SoftActorCritic::with_hidden_dim(4, 32)
        .with_train_schedule(4, 32)
        .with_gamma(0.95)
        .with_tau(0.05)
        .with_learning_rate(1e-3)
        .with_episode_layout(8, 32);
    agent
        .train_on_frame(&df, &labels)
        .expect("sac training should succeed on the fixture");
    agent
}

// ---------------------------------------------------------------------------
// (a) actor outputs a valid probability simplex
// ---------------------------------------------------------------------------

#[test]
fn actor_policy_is_a_valid_probability_simplex_inferbackend() {
    // Pure-tensor test on InferBackend (no autodiff) — proves the
    // softmax head always emits a normalized, non-negative 3-simplex.
    let device = <InferBackend as burn::tensor::backend::BackendTypes>::Device::default();
    let actor = SacNetConfig::new()
        .with_input_dim(4)
        .with_hidden_dim(16)
        .init_actor::<InferBackend>(&device);

    let states = Tensor::<InferBackend, 2>::from_data(
        TensorData::new(
            vec![
                0.5_f32, -1.0, 2.0, 0.3, //
                -2.0, 0.1, 0.0, 1.5, //
                10.0, -10.0, 5.0, -5.0, // extreme magnitudes
            ],
            [3, 4],
        ),
        &device,
    );
    let probs = actor
        .policy(states)
        .into_data()
        .to_vec::<f32>()
        .expect("policy probs");
    assert_eq!(probs.len(), 3 * NUM_ACTIONS);
    for row in probs.chunks_exact(NUM_ACTIONS) {
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "row must sum to 1, got {sum} ({row:?})"
        );
        for &p in row {
            assert!(p >= 0.0, "probabilities must be non-negative, got {p}");
            assert!(p.is_finite(), "probabilities must be finite, got {p}");
        }
    }
}

#[test]
fn trained_policy_probabilities_form_a_simplex() -> Result<()> {
    let agent = trained_agent(192);
    let probs = agent.policy_probabilities(&[0.2, -0.4, 0.6, 0.1])?;
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "policy must sum to 1, got {sum}");
    for &p in &probs {
        assert!(p >= 0.0 && p.is_finite(), "invalid probability {p}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// (b) one training step reduces critic loss / moves Q toward rewards
// ---------------------------------------------------------------------------

#[test]
fn update_step_reduces_critic_loss_on_a_tiny_fixture() {
    // Single fixed batch repeated: a stationary target. The full-
    // information critic regression should drive critic loss down.
    let mut agent = SoftActorCritic::with_hidden_dim(4, 32)
        .with_gamma(0.0) // no bootstrap → target == observed reward
        .with_tau(0.05)
        .with_learning_rate(5e-3);
    // bootstrap runtime identity (train_on_frame normally sets these).
    agent.feature_columns = vec![
        "f0".into(),
        "f1".into(),
        "f2".into(),
        "f3".into(),
    ];

    let batch: Vec<SacTuple> = (0..16)
        .map(|i| {
            let x = i as f32 * 0.1;
            SacTuple {
                state: vec![x, -x, 0.5 * x, 1.0 - x],
                next_state: vec![x + 0.1, -x - 0.1, 0.5 * x, 1.0 - x],
                // distinct per-action rewards so the critic has signal
                rewards: [0.2, 0.8, -0.5],
                done: true, // gamma=0 anyway, but keep targets == rewards
            }
        })
        .collect();

    let (loss_before, _, _) = agent
        .update_on_batch(&batch)
        .expect("first update should succeed");
    let mut loss_after = loss_before;
    for _ in 0..40 {
        let (loss, _, _) = agent
            .update_on_batch(&batch)
            .expect("update should succeed");
        loss_after = loss;
    }
    assert!(
        loss_after < loss_before,
        "critic loss should decrease: before={loss_before}, after={loss_after}"
    );
    // With gamma=0 and done=true the target is exactly the observed
    // reward vector — the critic should be able to drive loss low.
    assert!(
        loss_after < loss_before * 0.5,
        "critic loss should drop substantially: before={loss_before}, after={loss_after}"
    );
}

#[test]
fn q_values_move_toward_observed_rewards() -> Result<()> {
    // gamma=0, done=true => target y(s,a) = r(s,a). After many steps the
    // critic's Q(s,·) should approximate the per-action reward ordering.
    let mut agent = SoftActorCritic::with_hidden_dim(4, 32)
        .with_gamma(0.0)
        .with_tau(0.1)
        .with_learning_rate(5e-3);
    agent.feature_columns = vec!["f0".into(), "f1".into(), "f2".into(), "f3".into()];

    let state = vec![0.3_f32, -0.2, 0.7, 0.1];
    let rewards = [0.1_f32, 0.9, -0.6]; // Buy clearly best, Sell worst
    let batch: Vec<SacTuple> = (0..8)
        .map(|_| SacTuple {
            state: state.clone(),
            next_state: state.clone(),
            rewards,
            done: true,
        })
        .collect();
    for _ in 0..120 {
        agent.update_on_batch(&batch)?;
    }

    let device = agent.device.clone();
    let state_tensor = Tensor::<crate::burn_models::TrainBackend, 1>::from_data(
        TensorData::new(state.clone(), [4]),
        &device,
    )
    .unsqueeze::<2>();
    let q = super::SacCriticNet::forward(&agent.critic1, state_tensor)
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("read q: {err:?}"))?;
    assert_eq!(q.len(), NUM_ACTIONS);
    // The action ordering of Q should match the reward ordering:
    // Buy (idx 1) highest, Sell (idx 2) lowest.
    assert!(
        q[1] > q[0] && q[0] > q[2],
        "Q ordering should track rewards [hold<buy, sell<hold]: q={q:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// (c) artifact save -> load round-trips and reproduces identical probs
// ---------------------------------------------------------------------------

#[test]
fn save_load_round_trips_and_reproduces_inference() -> Result<()> {
    let agent = trained_agent(192);
    let path = unique_temp_dir("sac-roundtrip");
    agent.save(&path).expect("save should succeed");

    let loaded = SoftActorCritic::load(&path).expect("load should succeed");
    assert_eq!(loaded.state_dim, agent.state_dim);
    assert_eq!(loaded.feature_columns, agent.feature_columns);

    let state = [0.15_f32, -0.3, 0.55, 0.2];
    let before = agent.policy_probabilities(&state)?;
    let after = loaded.policy_probabilities(&state)?;
    for (b, a) in before.iter().zip(after.iter()) {
        assert!(
            (b - a).abs() < 1e-5,
            "round-tripped policy probs must match: {before:?} vs {after:?}"
        );
    }

    // metadata sidecar must exist and parse as the canonical 3-class map.
    assert!(path.join("config.json").exists());
    assert!(path.join(crate::statistical::common::METADATA_FILE_NAME).exists());

    let _ = std::fs::remove_dir_all(&path);
    Ok(())
}

#[test]
fn predict_runtime_emits_canonical_three_class_predictions() -> Result<()> {
    let agent = trained_agent(192);
    let (df, _) = fixture_frame(5);
    // predict_runtime requires the exact training feature schema.
    let preds = agent.predict_runtime(&df)?;
    assert_eq!(preds.len(), 5);
    for pred in &preds {
        let probs = pred.class_probabilities();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "runtime probs must sum to 1");
        assert!(pred.metadata().execution_backend.is_some());
    }
    Ok(())
}

#[test]
fn save_rejects_untrained_agent() {
    let agent = SoftActorCritic::with_hidden_dim(4, 16);
    let path = unique_temp_dir("sac-untrained");
    let err = agent
        .save(&path)
        .expect_err("untrained agent should not persist");
    assert!(
        err.to_string().contains("untrained runtime state"),
        "unexpected error: {err}"
    );
    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn predict_runtime_rejects_untrained_agent() {
    let agent = SoftActorCritic::with_hidden_dim(4, 16);
    let (df, _) = fixture_frame(3);
    let err = agent
        .predict_runtime(&df)
        .expect_err("cold sac should not run inference");
    assert!(
        err.to_string().contains("untrained runtime state"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// supporting-helper unit tests
// ---------------------------------------------------------------------------

#[test]
fn target_entropy_uses_scaled_log_num_actions() {
    let h = default_target_entropy(0.98);
    let expected = 0.98 * (NUM_ACTIONS as f32).ln();
    assert!((h - expected).abs() < 1e-6);
}

#[test]
fn tuples_from_episodes_rejects_dimension_mismatch() {
    let episode = crate::rl::TradingEpisode {
        transitions: vec![TradingTransition {
            state: vec![0.0, 1.0, 2.0],
            next_state: vec![0.0, 1.0], // wrong width
            rewards: [0.1, 0.2, 0.3],
            done: false,
        }],
    };
    let err = tuples_from_episodes(&[episode], 3).expect_err("dimension mismatch must fail");
    assert!(err.to_string().contains("state dimension mismatch"));
}

#[test]
fn average_rewards_computes_per_action_means() {
    let tuples = vec![
        SacTuple {
            state: vec![0.0; 4],
            next_state: vec![0.0; 4],
            rewards: [0.0, 1.0, -1.0],
            done: true,
        },
        SacTuple {
            state: vec![0.0; 4],
            next_state: vec![0.0; 4],
            rewards: [1.0, 1.0, -1.0],
            done: true,
        },
    ];
    let (hold, buy, sell) = average_rewards(&tuples);
    assert!((hold - 0.5).abs() < 1e-6);
    assert!((buy - 1.0).abs() < 1e-6);
    assert!((sell + 1.0).abs() < 1e-6);
}
