#![cfg(feature = "burn-cuda-backend")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use burn::prelude::Backend;
use burn_cuda::CudaDevice;
use burn_fusion::{inspect::FusionInspector, stream::StreamId};
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_models::base::ExpertModel;
use neoethos_models::burn_models::{InferBackend, burn_cuda_residency_scope, resolve_train_device};
use neoethos_models::deep_models::{BurnDeepExpert, DeepModelKind};
use neoethos_models::exit_agent::{ExitAgent, ExitAgentArtifact};
use neoethos_models::runtime::prediction::RuntimePrediction;
use neoethos_models::soft_actor_critic::{SoftActorCritic, SoftActorCriticArtifact};

const CUDA_ORDINAL: usize = 0;
const EXPLICIT_CUDA_POLICY: &str = "gpu:0";
const FEATURE_COUNT: usize = 4;
static NEXT_ARTIFACT_ID: AtomicU64 = AtomicU64::new(0);
static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

struct ArtifactDir(PathBuf);

impl ArtifactDir {
    fn unique(surface: &str) -> Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let id = NEXT_ARTIFACT_ID.fetch_add(1, Ordering::Relaxed);
        Ok(Self(std::env::temp_dir().join(format!(
            "neoethos-burn-cuda-{surface}-{}-{nonce}-{id}",
            std::process::id()
        ))))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ArtifactDir {
    fn drop(&mut self) {
        if self.0.exists() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn one_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(1).expect("one worker is valid");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("isolated Burn CUDA lifecycle lease")
}

fn exclusive_cuda_test() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .expect("another mandatory Burn CUDA gate panicked while holding the device")
}

fn typed_frame(rows: usize, offset: f64) -> Result<FeatureFrame> {
    let columns = (0..FEATURE_COUNT)
        .map(|column| {
            let values = (0..rows)
                .map(|row| {
                    offset
                        + column as f64 * 0.17
                        + row as f64 * 0.013
                        + ((row + column) % 3) as f64 * 0.021
                })
                .collect::<Vec<_>>();
            FeatureColumnF64::new(
                format!("cuda_feature_{column}"),
                values,
                vec![FeatureCellValidity::Valid; rows],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
}

fn three_class_labels(rows: usize) -> Vec<i32> {
    (0..rows)
        .map(|row| match row % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect()
}

fn assert_exact_cuda_device() -> Result<()> {
    let (device, selection) = resolve_train_device(EXPLICIT_CUDA_POLICY)?;
    assert_eq!(device.index, CUDA_ORDINAL);
    assert_eq!(selection.requested_policy, EXPLICIT_CUDA_POLICY);
    assert_eq!(selection.effective_policy, EXPLICIT_CUDA_POLICY);
    assert_eq!(selection.execution_backend, "cuda");
    Ok(())
}

fn assert_cuda_artifact_identity(requested: &str, effective: &str, backend: &str) {
    assert_eq!(requested, EXPLICIT_CUDA_POLICY);
    assert_eq!(effective, EXPLICIT_CUDA_POLICY);
    assert_eq!(backend, "cuda");
}

fn sync_and_assert_no_new_fusion_handles(inspector: &FusionInspector, phase: &str) -> Result<()> {
    <InferBackend as Backend>::sync(&CudaDevice::new(CUDA_ORDINAL))
        .with_context(|| format!("synchronize Burn Fusion after SAC {phase}"))?;
    let leaked_handles = inspector.new_handles_since_baseline();
    assert!(
        leaked_handles.is_empty(),
        "SAC {phase} retained {} CUDA Fusion handles: {leaked_handles:?}",
        leaked_handles.len()
    );
    Ok(())
}

fn assert_cuda_predictions(predictions: &[RuntimePrediction], expected_rows: usize) {
    assert_eq!(predictions.len(), expected_rows);
    for prediction in predictions {
        assert_eq!(
            prediction.metadata().execution_backend.as_deref(),
            Some("cuda")
        );
        assert_eq!(prediction.metadata().degraded_reason, None);
        let probabilities = prediction.class_probabilities();
        assert!(probabilities.iter().all(|value| value.is_finite()));
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() <= 1e-3);
    }
}

fn assert_prediction_parity(before: &[RuntimePrediction], after: &[RuntimePrediction]) {
    assert_eq!(before.len(), after.len());
    for (before, after) in before.iter().zip(after) {
        for (before, after) in before
            .class_probabilities()
            .into_iter()
            .zip(after.class_probabilities())
        {
            assert!(
                (before - after).abs() <= 1e-5,
                "Burn CUDA save/load probability drift: before={before}, after={after}"
            );
        }
    }
}

fn deep_cuda_params() -> HashMap<String, String> {
    HashMap::from([
        ("device".to_string(), EXPLICIT_CUDA_POLICY.to_string()),
        ("training_precision".to_string(), "fp32".to_string()),
        ("max_epochs".to_string(), "1".to_string()),
        ("batch_size".to_string(), "4".to_string()),
        ("patience".to_string(), "1".to_string()),
        ("hidden_dim".to_string(), "8".to_string()),
        ("n_layers".to_string(), "1".to_string()),
        ("n_blocks".to_string(), "1".to_string()),
        ("n_steps".to_string(), "1".to_string()),
        ("n_heads".to_string(), "1".to_string()),
        ("dim_ff".to_string(), "16".to_string()),
        ("patch_size".to_string(), "2".to_string()),
        ("n_periods".to_string(), "1".to_string()),
        ("token_count".to_string(), "2".to_string()),
        ("grid_size".to_string(), "3".to_string()),
        ("dropout".to_string(), "0".to_string()),
    ])
}

fn assert_deep_artifact_identity(path: &Path) -> Result<()> {
    let artifact: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path.join("config.json")).context("read deep CUDA artifact config")?,
    )
    .context("parse deep CUDA artifact config")?;
    let params = artifact
        .get("params")
        .and_then(serde_json::Value::as_object)
        .context("deep CUDA artifact is missing params")?;
    assert_cuda_artifact_identity(
        params
            .get("requested_device_policy")
            .and_then(serde_json::Value::as_str)
            .context("deep CUDA artifact is missing requested_device_policy")?,
        params
            .get("effective_device_policy")
            .and_then(serde_json::Value::as_str)
            .context("deep CUDA artifact is missing effective_device_policy")?,
        params
            .get("execution_backend")
            .and_then(serde_json::Value::as_str)
            .context("deep CUDA artifact is missing execution_backend")?,
    );
    let report = artifact
        .get("burn_training_report")
        .and_then(serde_json::Value::as_object)
        .context("deep CUDA artifact is missing Burn training report")?;
    assert_cuda_artifact_identity(
        report
            .get("requested_device_policy")
            .and_then(serde_json::Value::as_str)
            .context("deep CUDA report is missing requested_device_policy")?,
        report
            .get("effective_device_policy")
            .and_then(serde_json::Value::as_str)
            .context("deep CUDA report is missing effective_device_policy")?,
        report
            .get("execution_backend")
            .and_then(serde_json::Value::as_str)
            .context("deep CUDA report is missing execution_backend")?,
    );
    assert!(path.join("model.mpk").is_file());
    assert!(path.join("metadata.json").is_file());
    Ok(())
}

fn exercise_deep_cuda_lifecycle(kind: DeepModelKind) -> Result<()> {
    let _cuda_guard = exclusive_cuda_test();
    let _cuda_residency = burn_cuda_residency_scope(CUDA_ORDINAL);
    assert_exact_cuda_device()?;
    let train = typed_frame(12, 0.0)?;
    let validation = typed_frame(6, 1.0)?;
    let inference = typed_frame(4, 2.0)?;
    let train_labels = three_class_labels(12);
    let validation_labels = three_class_labels(6);
    let lease = one_worker_lease();

    let mut expert = BurnDeepExpert::new(kind, 41, Some(deep_cuda_params()));
    expert.fit_with_validation(
        &train,
        &train_labels,
        Some(&validation),
        Some(&validation_labels),
        &lease,
    )?;
    let before = expert.predict_runtime(&inference, &lease)?;
    assert_cuda_predictions(&before, inference.n_samples());

    let artifact_dir = ArtifactDir::unique(kind.model_name())?;
    expert.save(artifact_dir.path())?;
    assert_deep_artifact_identity(artifact_dir.path())?;

    let mut loaded = BurnDeepExpert::new(kind, 41, None);
    loaded.load(artifact_dir.path())?;
    let after = loaded.predict_runtime(&inference, &lease)?;
    assert_cuda_predictions(&after, inference.n_samples());
    assert_prediction_parity(&before, &after);
    Ok(())
}

macro_rules! deep_cuda_lifecycle_gate {
    ($name:ident, $kind:expr) => {
        #[test]
        fn $name() -> Result<()> {
            exercise_deep_cuda_lifecycle($kind)
        }
    };
}

deep_cuda_lifecycle_gate!(burn_cuda_deep_mlp_lifecycle_gpu0, DeepModelKind::Mlp);
deep_cuda_lifecycle_gate!(burn_cuda_deep_nbeats_lifecycle_gpu0, DeepModelKind::NBeats);
deep_cuda_lifecycle_gate!(
    burn_cuda_deep_nbeatsx_nf_lifecycle_gpu0,
    DeepModelKind::NBeatsxNf
);
deep_cuda_lifecycle_gate!(burn_cuda_deep_tide_lifecycle_gpu0, DeepModelKind::TiDE);
deep_cuda_lifecycle_gate!(burn_cuda_deep_tide_nf_lifecycle_gpu0, DeepModelKind::TiDENf);
deep_cuda_lifecycle_gate!(burn_cuda_deep_tabnet_lifecycle_gpu0, DeepModelKind::TabNet);
deep_cuda_lifecycle_gate!(burn_cuda_deep_kan_lifecycle_gpu0, DeepModelKind::Kan);
deep_cuda_lifecycle_gate!(
    burn_cuda_deep_transformer_lifecycle_gpu0,
    DeepModelKind::Transformer
);
deep_cuda_lifecycle_gate!(
    burn_cuda_deep_patchtst_lifecycle_gpu0,
    DeepModelKind::PatchTst
);
deep_cuda_lifecycle_gate!(
    burn_cuda_deep_timesnet_lifecycle_gpu0,
    DeepModelKind::TimesNet
);

#[test]
fn burn_cuda_exit_agent_lifecycle_gpu0() -> Result<()> {
    let _cuda_guard = exclusive_cuda_test();
    let _cuda_residency = burn_cuda_residency_scope(CUDA_ORDINAL);
    assert_exact_cuda_device()?;
    let frame = typed_frame(64, 3.0)?;
    let labels = three_class_labels(frame.n_samples());
    let lease = one_worker_lease();
    let mut agent = ExitAgent::with_hidden_dim(FEATURE_COUNT, 8)
        .with_device_policy(EXPLICIT_CUDA_POLICY)?
        .with_reward_horizon(2)
        .with_warmup_steps(8);
    let report = agent.fit_from_frame_with_report(&frame, &labels, &lease)?;
    assert_cuda_artifact_identity(
        &report.requested_device_policy,
        &report.effective_device_policy,
        &report.execution_backend,
    );
    let before = agent.predict_runtime(&frame, &lease)?;
    assert_cuda_predictions(&before, frame.n_samples());

    let artifact_dir = ArtifactDir::unique("exit-agent")?;
    agent.save(artifact_dir.path())?;
    let artifact: ExitAgentArtifact = serde_json::from_slice(
        &std::fs::read(artifact_dir.path().join("config.json"))
            .context("read ExitAgent CUDA artifact")?,
    )
    .context("parse ExitAgent CUDA artifact")?;
    assert_cuda_artifact_identity(
        artifact
            .requested_device_policy
            .as_deref()
            .context("ExitAgent artifact is missing requested_device_policy")?,
        artifact
            .effective_device_policy
            .as_deref()
            .context("ExitAgent artifact is missing effective_device_policy")?,
        artifact
            .execution_backend
            .as_deref()
            .context("ExitAgent artifact is missing execution_backend")?,
    );

    let loaded = ExitAgent::load(artifact_dir.path())?;
    let after = loaded.predict_runtime(&frame, &lease)?;
    assert_cuda_predictions(&after, frame.n_samples());
    assert_prediction_parity(&before, &after);
    Ok(())
}

#[test]
fn burn_cuda_sac_lifecycle_gpu0() -> Result<()> {
    let _cuda_guard = exclusive_cuda_test();
    let _cuda_residency = burn_cuda_residency_scope(CUDA_ORDINAL);
    let fusion_inspector = FusionInspector::install(StreamId::current());
    <InferBackend as Backend>::sync(&CudaDevice::new(CUDA_ORDINAL))
        .context("synchronize Burn Fusion before the SAC phase baseline")?;
    fusion_inspector.set_baseline();
    assert_exact_cuda_device()?;
    let frame = typed_frame(64, 4.0)?;
    let labels = three_class_labels(frame.n_samples());
    let lease = one_worker_lease();

    {
        let mut training_probe = SoftActorCritic::with_hidden_dim(FEATURE_COUNT, 8)
            .with_device_policy(EXPLICIT_CUDA_POLICY)?
            .with_train_schedule(1, 8)
            .with_episode_layout(2, 8);
        training_probe.train_on_frame(&frame, &labels, &lease)?;
    }
    sync_and_assert_no_new_fusion_handles(&fusion_inspector, "training teardown")?;

    let mut agent = SoftActorCritic::with_hidden_dim(FEATURE_COUNT, 8)
        .with_device_policy(EXPLICIT_CUDA_POLICY)?
        .with_train_schedule(1, 8)
        .with_episode_layout(2, 8);
    let report = agent.train_on_frame(&frame, &labels, &lease)?;
    assert_cuda_artifact_identity(
        &report.requested_device_policy,
        &report.effective_device_policy,
        &report.execution_backend,
    );
    <InferBackend as Backend>::sync(&CudaDevice::new(CUDA_ORDINAL))
        .context("synchronize Burn Fusion before the first SAC inference baseline")?;
    fusion_inspector.set_baseline();
    let before = agent.predict_runtime(&frame, &lease)?;
    assert_cuda_predictions(&before, frame.n_samples());
    sync_and_assert_no_new_fusion_handles(&fusion_inspector, "first inference")?;

    let artifact_dir = ArtifactDir::unique("sac")?;
    agent.save(artifact_dir.path())?;
    let artifact: SoftActorCriticArtifact = serde_json::from_slice(
        &std::fs::read(artifact_dir.path().join("config.json"))
            .context("read SAC CUDA artifact")?,
    )
    .context("parse SAC CUDA artifact")?;
    assert_cuda_artifact_identity(
        artifact
            .requested_device_policy
            .as_deref()
            .context("SAC artifact is missing requested_device_policy")?,
        artifact
            .effective_device_policy
            .as_deref()
            .context("SAC artifact is missing effective_device_policy")?,
        artifact
            .execution_backend
            .as_deref()
            .context("SAC artifact is missing execution_backend")?,
    );

    let loaded = SoftActorCritic::load(artifact_dir.path())?;
    <InferBackend as Backend>::sync(&CudaDevice::new(CUDA_ORDINAL))
        .context("synchronize Burn Fusion before the reloaded SAC inference baseline")?;
    fusion_inspector.set_baseline();
    let after = loaded.predict_runtime(&frame, &lease)?;
    assert_cuda_predictions(&after, frame.n_samples());
    sync_and_assert_no_new_fusion_handles(&fusion_inspector, "reloaded inference")?;
    assert_prediction_parity(&before, &after);
    Ok(())
}
