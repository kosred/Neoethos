#![cfg(feature = "statistical-gpu")]

use anyhow::{Context, Result, bail};
use ndarray::Array2;
use neoethos_core::Settings;
use neoethos_data::{FeatureFrame, test_fixtures};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_models::base::ExpertModel;
use neoethos_models::runtime::prediction::RuntimePrediction;
use neoethos_models::statistical::common::install_statistical_runtime_from_settings;
use neoethos_models::streaming::OnlinePassiveAggressiveExpert;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const RTX_CHILD_ROLE_ENV: &str = "NEOETHOS_ONLINE_PA_RTX_CHILD_ROLE_TEST_ONLY";
const RTX_CHILD_RECEIPT_ENV: &str = "NEOETHOS_ONLINE_PA_RTX_CHILD_RECEIPT_TEST_ONLY";
const RTX_CHILD_SEQUENCE_ENV: &str = "NEOETHOS_ONLINE_PA_RTX_CHILD_SEQUENCE_TEST_ONLY";
const CPU_WORKER_WIDTH: usize = 7;
const ONLINE_PA_SEARCH_MAX_EPOCHS: usize = 12;
const LIFECYCLE_WALL_TIMEOUT: Duration = Duration::from_secs(20 * 60);
const PARITY_WALL_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const MAX_EPOCH_WALL_TIMEOUT: Duration = Duration::from_secs(20 * 60);
const BENCH_WALL_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const PB_V2_SCHEMA: &str =
    "neoethos.online_pa.prediction_based.class_weighted_slack_cap.bias_augmented.f64.v2";

fn seven_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(CPU_WORKER_WIDTH).expect("seven workers are valid");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("online_pa hardware test can acquire its seven-worker production budget")
}

fn install_device_policy(policy: &str) {
    let mut settings = Settings::default();
    settings.system.device = policy.to_string();
    settings.models.statistical_device = policy.to_string();
    settings
        .models
        .model_param_overrides
        .entry("online_pa".to_string())
        .or_default()
        .insert("device".to_string(), policy.to_string());
    install_statistical_runtime_from_settings(&settings);
}

fn deterministic_matrix(rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        let intraday = (row % 86_400) as f64 / 86_400.0;
        let microstructure = ((row * 17 + col * 131) % 8192) as f64 / 4096.0 - 1.0;
        let class_signal = if col % 3 == row % 3 { 2.0 } else { -1.0 };
        class_signal + intraday * 0.05 + microstructure * 0.02 + col as f64 * 1.0e-4
    })
}

fn deterministic_frame(rows: usize, cols: usize) -> Result<FeatureFrame> {
    let names = (0..cols)
        .map(|col| format!("online_pa_feature_{col}"))
        .collect();
    test_fixtures::ctrader_test_feature_frame_from_matrix(
        test_fixtures::canonical_test_timestamps(rows),
        names,
        deterministic_matrix(rows, cols),
    )
}

fn labels(rows: usize) -> Vec<i32> {
    (0..rows)
        .map(|row| match row % 3 {
            0 => 0,
            1 => 1,
            _ => -1,
        })
        .collect()
}

struct TempArtifactDir(PathBuf);

impl TempArtifactDir {
    fn create(prefix: &str) -> Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system time before Unix epoch")?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("neoethos-{prefix}-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path)
            .with_context(|| format!("create test artifact directory {}", path.display()))?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempArtifactDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!("failed to remove {}: {error}", self.0.display());
        }
    }
}

fn assert_probability_rows_equal(left: &Array2<f64>, right: &Array2<f64>) {
    assert_eq!(left.dim(), right.dim());
    for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "probability {index} drifted"
        );
    }
}

fn assert_runtime_is_proven_cuda_without_degradation(runtime: &[RuntimePrediction]) {
    for prediction in runtime {
        let metadata = prediction.metadata();
        assert!(
            metadata
                .execution_backend
                .as_deref()
                .is_some_and(|backend| backend.contains("cuda")),
            "runtime metadata did not prove CUDA inference: {metadata:?}"
        );
        assert_eq!(
            metadata.degraded_reason, None,
            "a proven CUDA artifact cannot carry CPU fallback/degradation metadata"
        );
    }
}

fn ndarray_json_data_mut<'a>(artifact: &'a mut Value, field: &str) -> Result<&'a mut Vec<Value>> {
    let value = artifact
        .get_mut(field)
        .with_context(|| format!("artifact field `{field}`"))?;
    if value.is_array() {
        return value
            .as_array_mut()
            .with_context(|| format!("artifact array field `{field}`"));
    }
    value
        .get_mut("data")
        .and_then(Value::as_array_mut)
        .with_context(|| format!("ndarray artifact data `{field}.data`"))
}

fn exercise_full_cuda_lifecycle(requested_policy: &str) -> Result<Value> {
    install_device_policy(requested_policy);
    let frame = deterministic_frame(4096, 64)?;
    let labels = labels(frame.n_samples());
    let lease = seven_worker_lease();
    let mut fitted = OnlinePassiveAggressiveExpert::new(1.0, 2);
    fitted.fit(&frame, &labels, &lease)?;
    let before = fitted.predict_proba(&frame, &lease)?;
    let runtime = fitted.predict_runtime(&frame, &lease)?;
    assert_eq!(runtime.len(), frame.n_samples());
    assert_runtime_is_proven_cuda_without_degradation(&runtime);

    let artifact_dir = TempArtifactDir::create("online-pa-full-gpu-lifecycle")?;
    fitted.save(artifact_dir.path())?;
    let model_path = artifact_dir.path().join("model.json");
    let artifact: Value = serde_json::from_slice(&std::fs::read(&model_path)?)?;
    let requested = artifact
        .pointer("/cuda_training_evidence/full_pipeline/requested_device_policy")
        .and_then(Value::as_str)
        .context("persisted requested device policy")?;
    let effective = artifact
        .pointer("/cuda_training_evidence/full_pipeline/effective_device_policy")
        .and_then(Value::as_str)
        .context("persisted effective device policy")?;
    assert_eq!(requested, requested_policy);
    let ordinal = effective
        .strip_prefix("gpu:")
        .context("requested policy did not resolve to exact gpu:N")?
        .parse::<u32>()?;
    if requested_policy == "gpu:0" {
        assert_eq!(effective, "gpu:0");
        assert_eq!(ordinal, 0);
    }
    let identity = artifact
        .pointer("/cuda_training_evidence/full_pipeline/device_identity")
        .context("persisted physical CUDA identity")?;
    assert_eq!(identity["ordinal"], ordinal);
    assert!(
        identity["name"]
            .as_str()
            .is_some_and(|name| !name.trim().is_empty())
    );
    assert!(identity["uuid"].as_array().is_some_and(|uuid| {
        uuid.len() == 16
            && uuid
                .iter()
                .any(|byte| byte.as_u64().unwrap_or_default() != 0)
    }));
    assert!(identity["total_memory_bytes"].as_u64().unwrap_or_default() > 0);
    let device_uuid = identity["uuid"].clone();
    let effective_device_policy = effective.to_string();

    let mut loaded = OnlinePassiveAggressiveExpert::new(0.25, 1);
    loaded.load(artifact_dir.path())?;
    let after = loaded.predict_proba(&frame, &lease)?;
    assert_probability_rows_equal(&before, &after);
    let loaded_runtime = loaded.predict_runtime(&frame, &lease)?;
    assert_runtime_is_proven_cuda_without_degradation(&loaded_runtime);

    let missing = deterministic_frame(64, 63)?;
    let missing_error = loaded
        .predict_proba(&missing, &lease)
        .expect_err("missing feature columns must fail before inference");
    assert!(format!("{missing_error:#}").contains("feature"));

    let mut arithmetic_fault_artifact = artifact.clone();
    let bias = ndarray_json_data_mut(&mut arithmetic_fault_artifact, "bias")?;
    anyhow::ensure!(
        bias.len() == 3,
        "online_pa artifact bias must have three entries"
    );
    bias[0] = json!(f64::MAX);
    bias[1] = json!(-f64::MAX);
    bias[2] = json!(0.0);
    std::fs::write(
        &model_path,
        serde_json::to_vec_pretty(&arithmetic_fault_artifact)?,
    )?;
    let mut arithmetic_fault = OnlinePassiveAggressiveExpert::new(1.0, 1);
    arithmetic_fault.load(artifact_dir.path())?;
    let arithmetic_error = arithmetic_fault
        .predict_proba(&frame.row_window(0, 1)?, &lease)
        .expect_err("[MAX,-MAX,0] artifact logits must fail before exp");
    assert!(
        format!("{arithmetic_error:#}").contains("device fault code 50"),
        "expected DEVICE_INFERENCE_ARITHMETIC_FAULT, got {arithmetic_error:#}"
    );

    let mut identity_fault_artifact = artifact;
    let uuid = identity_fault_artifact
        .pointer_mut("/cuda_training_evidence/full_pipeline/device_identity/uuid")
        .and_then(Value::as_array_mut)
        .context("mutable persisted CUDA UUID")?;
    let original = uuid[0].as_u64().context("UUID byte")?;
    uuid[0] = json!((original as u8) ^ 0xff);
    std::fs::write(
        &model_path,
        serde_json::to_vec_pretty(&identity_fault_artifact)?,
    )?;
    let mut rejected = OnlinePassiveAggressiveExpert::new(1.0, 1);
    let identity_error = rejected
        .load(artifact_dir.path())
        .expect_err("fresh load must reject a mutated physical identity");
    assert!(format!("{identity_error:#}").contains("physical device identity"));
    Ok(json!({
        "schema": "neoethos.online_pa.full_cuda_lifecycle.v1",
        "requested_device_policy": requested_policy,
        "effective_device_policy": effective_device_policy,
        "device_uuid": device_uuid,
        "rows": frame.n_samples(),
        "cols": 64,
    }))
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

#[derive(Debug)]
struct TimedProbabilitySample {
    probabilities: Array2<f64>,
    elapsed: Duration,
}

fn weighted_probability_checksum(probabilities: &Array2<f64>) -> f64 {
    probabilities
        .iter()
        .copied()
        .enumerate()
        .map(|(index, probability)| probability * ((index % 17) + 1) as f64)
        .sum::<f64>()
        / probabilities.nrows() as f64
}

fn pb_v2_cpu_oracle(
    raw: &Array2<f64>,
    original_labels: &[i32],
    aggressiveness: f64,
    epochs: usize,
    pool: &rayon::ThreadPool,
) -> Result<Array2<f64>> {
    const ROWS_PER_SCALER_PARTIAL: usize = 1_024;
    let rows = raw.nrows();
    let cols = raw.ncols();
    anyhow::ensure!(
        rows > 0 && cols > 0,
        "PB-v2 oracle requires non-empty input"
    );
    anyhow::ensure!(original_labels.len() == rows, "PB-v2 oracle label mismatch");
    anyhow::ensure!(
        raw.iter().all(|value| value.is_finite()),
        "PB-v2 oracle input"
    );

    let mapped = original_labels
        .iter()
        .map(|label| match *label {
            0 => Ok(0_usize),
            1 => Ok(1_usize),
            -1 => Ok(2_usize),
            other => bail!("PB-v2 oracle invalid original label {other}"),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut class_counts = [0_usize; 3];
    for target in &mapped {
        class_counts[*target] += 1;
    }
    anyhow::ensure!(
        class_counts.iter().all(|count| *count > 0),
        "PB-v2 oracle requires all three classes"
    );
    let class_weights =
        class_counts.map(|count| (rows as f64 / (3.0 * count as f64)).clamp(0.5, 4.0));

    // The same ordered 1,024-row Welford partials and Chan finalize used by
    // the CUDA preprocessing path. Columns are independent and use all seven
    // budgeted CPU workers without changing any per-column operation order.
    let scaler = pool.install(|| {
        (0..cols)
            .into_par_iter()
            .map(|col| {
                let mut count = 0_u64;
                let mut mean = 0.0_f64;
                let mut m2 = 0.0_f64;
                for chunk_start in (0..rows).step_by(ROWS_PER_SCALER_PARTIAL) {
                    let chunk_end = (chunk_start + ROWS_PER_SCALER_PARTIAL).min(rows);
                    let mut partial_count = 0_u64;
                    let mut partial_mean = 0.0_f64;
                    let mut partial_m2 = 0.0_f64;
                    for row in chunk_start..chunk_end {
                        let value = raw[(row, col)];
                        partial_count += 1;
                        let delta = value - partial_mean;
                        partial_mean += delta / partial_count as f64;
                        partial_m2 += delta * (value - partial_mean);
                    }
                    let combined_count = count + partial_count;
                    let delta = partial_mean - mean;
                    let next_mean = mean + delta * partial_count as f64 / combined_count as f64;
                    let cross = delta * delta * (count as f64 * partial_count as f64)
                        / combined_count as f64;
                    mean = next_mean;
                    m2 += partial_m2 + cross;
                    count = combined_count;
                }
                let std = (m2 / count as f64).sqrt();
                (mean, if std > 1.0e-12 { std } else { 1.0 })
            })
            .collect::<Vec<_>>()
    });
    anyhow::ensure!(
        scaler
            .iter()
            .all(|(mean, std)| mean.is_finite() && std.is_finite() && *std > 0.0),
        "PB-v2 oracle scaler arithmetic fault"
    );
    let scaled = pool.install(|| {
        (0..raw.len())
            .into_par_iter()
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                (raw[(row, col)] - scaler[col].0) / scaler[col].1
            })
            .collect::<Vec<_>>()
    });
    anyhow::ensure!(
        scaled.iter().all(|value| value.is_finite()),
        "PB-v2 transform"
    );
    let scaled = Array2::from_shape_vec((rows, cols), scaled)?;

    // PA is order-dependent, so this loop intentionally stays epoch-major and
    // row-major. The seven-worker budget accelerates independent preprocessing
    // and inference, while preserving the exact sequential PB-v2 update.
    let mut weights = Array2::<f64>::zeros((3, cols));
    let mut bias = [0.0_f64; 3];
    for _epoch in 0..epochs {
        for (row, target) in mapped.iter().copied().enumerate() {
            let x = scaled.row(row);
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
            anyhow::ensure!(scores.iter().all(|score| score.is_finite()), "PB-v2 scores");
            let mut predicted = 0;
            if scores[1] >= scores[predicted] {
                predicted = 1;
            }
            if scores[2] >= scores[predicted] {
                predicted = 2;
            }
            if predicted == target {
                continue;
            }
            let margin = scores[predicted] - scores[target] + 1.0;
            if margin <= 0.0 {
                continue;
            }
            let tau = (margin / (2.0 * norm)).min(aggressiveness * class_weights[target]);
            anyhow::ensure!(tau.is_finite() && tau >= 0.0, "PB-v2 tau");
            for col in 0..cols {
                let delta = tau * x[col];
                weights[(target, col)] += delta;
                weights[(predicted, col)] -= delta;
            }
            bias[target] += tau;
            bias[predicted] -= tau;
        }
    }

    let probability_rows = pool.install(|| {
        (0..rows)
            .into_par_iter()
            .map(|row| -> Result<[f64; 3]> {
                let x = scaled.row(row);
                let mut logits = [0.0_f64; 3];
                for class in 0..3 {
                    logits[class] = weights
                        .row(class)
                        .iter()
                        .zip(x.iter())
                        .map(|(weight, value)| weight * value)
                        .sum::<f64>()
                        + bias[class];
                }
                anyhow::ensure!(logits.iter().all(|logit| logit.is_finite()), "PB-v2 logits");
                let maximum = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
                let shifted = logits.map(|logit| logit - maximum);
                anyhow::ensure!(
                    shifted.iter().all(|value| value.is_finite()),
                    "PB-v2 shifted logits"
                );
                let exps = shifted.map(f64::exp);
                let normalizer = exps.iter().sum::<f64>();
                anyhow::ensure!(normalizer.is_finite() && normalizer > 0.0, "PB-v2 softmax");
                Ok(exps.map(|value| value / normalizer))
            })
            .collect::<Vec<_>>()
    });
    let probability_rows = probability_rows.into_iter().collect::<Result<Vec<_>>>()?;
    let probabilities = probability_rows.into_iter().flatten().collect::<Vec<_>>();
    Array2::from_shape_vec((rows, 3), probabilities).context("shape PB-v2 CPU probabilities")
}

fn timed_pb_v2_cpu_sample(
    raw: &Array2<f64>,
    original_labels: &[i32],
    lease: &CpuLease,
    pool: &rayon::ThreadPool,
) -> Result<TimedProbabilitySample> {
    // ONLINE_PA_CPU7_TIMED_WORKLOAD_BEGIN
    let started = Instant::now();
    let probabilities = lease.scope(|| pb_v2_cpu_oracle(raw, original_labels, 1.0, 1, pool))?;
    let elapsed = started.elapsed();
    // ONLINE_PA_CPU7_TIMED_WORKLOAD_END
    Ok(TimedProbabilitySample {
        probabilities,
        elapsed,
    })
}

fn timed_expert_model_sample(
    frame: &FeatureFrame,
    original_labels: &[i32],
    lease: &CpuLease,
) -> Result<(TimedProbabilitySample, OnlinePassiveAggressiveExpert)> {
    // ONLINE_PA_EXPERT_TIMED_WORKLOAD_BEGIN
    let started = Instant::now();
    let mut model = OnlinePassiveAggressiveExpert::new(1.0, 1);
    model.fit(frame, original_labels, lease)?;
    let probabilities = model.predict_proba(frame, lease)?;
    let elapsed = started.elapsed();
    // ONLINE_PA_EXPERT_TIMED_WORKLOAD_END
    Ok((
        TimedProbabilitySample {
            probabilities,
            elapsed,
        },
        model,
    ))
}

fn run_benchmark_role(child_role: &str) -> Result<Value> {
    let (role, policy) = match child_role {
        "benchmark-pb-v2-cpu7" => ("pb_v2_cpu_oracle", "cpu"),
        "benchmark-gpu" => ("gpu", "auto"),
        "benchmark-legacy-cpu" => ("legacy_cpu", "cpu"),
        other => bail!("invalid benchmark child role `{other}`"),
    };
    install_device_policy(policy);
    let lease = seven_worker_lease();
    let oracle_pool = if role == "pb_v2_cpu_oracle" {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(CPU_WORKER_WIDTH)
                .thread_name(|index| format!("online-pa-pbv2-oracle-{index}"))
                .build()?,
        )
    } else {
        None
    };
    if let Some(pool) = &oracle_pool {
        anyhow::ensure!(
            pool.current_num_threads() == CPU_WORKER_WIDTH,
            "PB-v2 oracle did not receive seven Rayon workers"
        );
    }
    let mut widths = Vec::new();
    for cols in [64_usize, 128] {
        const ROWS: usize = 1_000_000;
        let original_labels = labels(ROWS);
        let raw = (role == "pb_v2_cpu_oracle").then(|| deterministic_matrix(ROWS, cols));
        let frame = if role == "pb_v2_cpu_oracle" {
            None
        } else {
            Some(deterministic_frame(ROWS, cols)?)
        };
        let mut samples = Vec::new();
        let mut checksum = 0.0_f64;
        let mut last_model = None;
        for _ in 0..5 {
            match role {
                "pb_v2_cpu_oracle" => {
                    let sample = timed_pb_v2_cpu_sample(
                        raw.as_ref().context("PB-v2 raw benchmark input")?,
                        &original_labels,
                        &lease,
                        oracle_pool.as_ref().context("PB-v2 seven-worker pool")?,
                    )?;
                    samples.push(sample.elapsed);
                    checksum = weighted_probability_checksum(&sample.probabilities);
                }
                "gpu" | "legacy_cpu" => {
                    let frame = frame.as_ref().context("production benchmark frame")?;
                    let (sample, model) =
                        timed_expert_model_sample(frame, &original_labels, &lease)?;
                    samples.push(sample.elapsed);
                    checksum = weighted_probability_checksum(&sample.probabilities);
                    last_model = Some(model);
                }
                _ => unreachable!("role validated above"),
            }
        }
        let (
            execution_backend,
            degraded_reason,
            training_semantics_schema,
            effective_device_policy,
            device_uuid,
        ) = if role == "pb_v2_cpu_oracle" {
            (
                "online_pa_pb_v2_cpu_oracle_test_only".to_string(),
                None,
                PB_V2_SCHEMA.to_string(),
                "cpu".to_string(),
                Value::Null,
            )
        } else {
            let model = last_model.context("benchmark produced no fitted model")?;
            let frame = frame.as_ref().context("production benchmark frame")?;
            let probe = frame.row_window(0, 1)?;
            let runtime = model.predict_runtime(&probe, &lease)?;
            let metadata = runtime[0].metadata();
            let execution_backend = metadata
                .execution_backend
                .as_deref()
                .context("benchmark runtime omitted execution backend")?
                .to_string();
            anyhow::ensure!(
                metadata.degraded_reason.is_none(),
                "benchmark route degraded or fell back: {:?}",
                metadata.degraded_reason
            );
            match role {
                "gpu" => anyhow::ensure!(
                    execution_backend.contains("cuda"),
                    "GPU benchmark child did not execute CUDA: {execution_backend}"
                ),
                "legacy_cpu" => anyhow::ensure!(
                    execution_backend == "online_pa_cpu",
                    "legacy CPU child used unexpected backend: {execution_backend}"
                ),
                _ => unreachable!("role validated above"),
            }
            let artifact_dir = TempArtifactDir::create("online-pa-benchmark-artifact")?;
            model.save(artifact_dir.path())?;
            let artifact: Value =
                serde_json::from_slice(&std::fs::read(artifact_dir.path().join("model.json"))?)?;
            let schema = artifact["training_semantics_schema"]
                .as_str()
                .context("benchmark artifact training semantics")?
                .to_string();
            let effective_device_policy = artifact["effective_device_policy"]
                .as_str()
                .context("benchmark artifact effective device policy")?
                .to_string();
            let device_uuid = artifact
                .pointer("/cuda_training_evidence/full_pipeline/device_identity/uuid")
                .cloned()
                .unwrap_or(Value::Null);
            (
                execution_backend,
                metadata.degraded_reason.as_ref().map(ToString::to_string),
                schema,
                effective_device_policy,
                device_uuid,
            )
        };
        let median = median(samples);
        widths.push(json!({
            "rows": ROWS,
            "cols": cols,
            "epochs": 1,
            "median_ns": median.as_nanos().to_string(),
            "weighted_probability_checksum": checksum,
            "execution_backend": execution_backend,
            "degraded_reason": degraded_reason,
            "training_semantics_schema": training_semantics_schema,
            "effective_device_policy": effective_device_policy,
            "device_uuid": device_uuid,
        }));
    }
    let effective_device_policy = widths[0]["effective_device_policy"].clone();
    let device_uuid = widths[0]["device_uuid"].clone();
    let receipt = json!({
        "schema": "neoethos.online_pa.same_pb_v2_seven_cpu_budget_benchmark.v2",
        "role": child_role,
        "benchmark_role": role,
        "policy": policy,
        "iterations": 5,
        "worker_width": CPU_WORKER_WIDTH,
        "rayon_worker_width": oracle_pool.as_ref().map(|pool| pool.current_num_threads()),
        "effective_device_policy": effective_device_policy,
        "device_uuid": device_uuid,
        "widths": widths,
    });
    Ok(receipt)
}

fn wait_child_with_timeout(mut child: Child, timeout: Duration, label: &str) -> Result<()> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().with_context(|| format!("poll {label}"))? {
            anyhow::ensure!(status.success(), "{label} failed: {status}");
            return Ok(());
        }
        if started.elapsed() >= timeout {
            child
                .kill()
                .with_context(|| format!("kill timed-out {label}"))?;
            let _ = child.wait();
            bail!("{label} exceeded wall timeout {timeout:?}");
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn validate_benchmark_gate(cpu: &Value, gpu: &Value, legacy: &Value) -> Result<()> {
    assert_eq!(cpu["schema"], gpu["schema"]);
    assert_eq!(cpu["schema"], legacy["schema"]);
    assert_eq!(cpu["worker_width"], CPU_WORKER_WIDTH);
    assert_eq!(gpu["worker_width"], CPU_WORKER_WIDTH);
    assert_eq!(legacy["worker_width"], CPU_WORKER_WIDTH);
    assert_eq!(cpu["rayon_worker_width"], CPU_WORKER_WIDTH);
    for index in 0..2 {
        let cpu_width = &cpu["widths"][index];
        let gpu_width = &gpu["widths"][index];
        let legacy_width = &legacy["widths"][index];
        assert_eq!(cpu_width["rows"], gpu_width["rows"]);
        assert_eq!(cpu_width["cols"], gpu_width["cols"]);
        assert_eq!(cpu_width["rows"], legacy_width["rows"]);
        assert_eq!(cpu_width["cols"], legacy_width["cols"]);
        assert_eq!(cpu_width["training_semantics_schema"], PB_V2_SCHEMA);
        assert_eq!(gpu_width["training_semantics_schema"], PB_V2_SCHEMA);
        assert_ne!(legacy_width["training_semantics_schema"], PB_V2_SCHEMA);
        assert_eq!(
            gpu_width["device_uuid"], gpu["device_uuid"],
            "every GPU benchmark width must bind the same physical CUDA UUID"
        );
        let cpu_checksum = cpu_width["weighted_probability_checksum"]
            .as_f64()
            .context("CPU PB-v2 checksum")?;
        let gpu_checksum = gpu_width["weighted_probability_checksum"]
            .as_f64()
            .context("GPU PB-v2 checksum")?;
        assert!(
            (cpu_checksum - gpu_checksum).abs() <= 2.0e-5,
            "{}x{} PB-v2 checksum drift: CPU {cpu_checksum}, GPU {gpu_checksum}",
            cpu_width["rows"],
            cpu_width["cols"],
        );
        let cpu_ns = cpu_width["median_ns"]
            .as_str()
            .context("CPU median_ns string")?
            .parse::<u128>()?;
        let gpu_ns = gpu_width["median_ns"]
            .as_str()
            .context("GPU median_ns string")?
            .parse::<u128>()?;
        assert!(
            gpu_ns.saturating_mul(5) < cpu_ns.saturating_mul(4),
            "{}x{} production GPU {gpu_ns}ns must be >=1.25x faster than CPU {cpu_ns}ns",
            cpu_width["rows"],
            cpu_width["cols"],
        );
    }
    // Legacy CPU timing is intentionally secondary: its persisted schema uses
    // the older implicit loss multiplier and is not the same PB-v2 workload.
    Ok(())
}

fn run_max_epoch_role(cols: usize) -> Result<Value> {
    anyhow::ensure!(
        matches!(cols, 64 | 128),
        "unsupported max-epoch width {cols}"
    );
    install_device_policy("auto");
    let frame = deterministic_frame(1_000_000, cols)?;
    let original_labels = labels(frame.n_samples());
    let lease = seven_worker_lease();
    let mut model = OnlinePassiveAggressiveExpert::new(1.0, ONLINE_PA_SEARCH_MAX_EPOCHS);
    let started = Instant::now();
    model.fit(&frame, &original_labels, &lease)?;
    let elapsed = started.elapsed();
    let artifact_dir = TempArtifactDir::create("online-pa-max-epoch")?;
    model.save(artifact_dir.path())?;
    let artifact: Value =
        serde_json::from_slice(&std::fs::read(artifact_dir.path().join("model.json"))?)?;
    let pipeline = artifact
        .pointer("/cuda_training_evidence/full_pipeline")
        .context("max-epoch full CUDA receipt")?;
    let chunks_per_epoch = 1_000_000_usize.div_ceil(1_024) as u64;
    assert_eq!(pipeline["training_rows_per_launch"], 1_024);
    assert_eq!(
        pipeline["training_row_chunk_count_per_epoch"],
        chunks_per_epoch
    );
    assert_eq!(
        pipeline["training_epoch_count"],
        ONLINE_PA_SEARCH_MAX_EPOCHS
    );
    assert_eq!(
        pipeline["training_launch_count"],
        chunks_per_epoch * ONLINE_PA_SEARCH_MAX_EPOCHS as u64
    );
    assert_eq!(pipeline["training_interchunk_device_to_host_bytes"], 0);
    let receipt = json!({
        "schema": "neoethos.online_pa.max_epoch_million_row_timeout.v1",
        "rows": 1_000_000,
        "cols": cols,
        "epochs": ONLINE_PA_SEARCH_MAX_EPOCHS,
        "worker_width": CPU_WORKER_WIDTH,
        "elapsed_ns": elapsed.as_nanos().to_string(),
        "training_launch_count": pipeline["training_launch_count"],
        "effective_device_policy": pipeline["effective_device_policy"],
        "device_uuid": pipeline["device_identity"]["uuid"],
    });
    Ok(receipt)
}

fn assert_probability_rows_close(left: &Array2<f64>, right: &Array2<f64>, tolerance: f64) {
    assert_eq!(left.dim(), right.dim());
    for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "probability {index}: CPU {left:.17e} drifted from GPU {right:.17e}"
        );
    }
}

fn run_million_row_parity_role(cols: usize) -> Result<Value> {
    const ROWS: usize = 1_000_000;
    anyhow::ensure!(matches!(cols, 64 | 128), "unsupported parity width {cols}");
    let original_labels = labels(ROWS);
    let lease = seven_worker_lease();
    let oracle_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(CPU_WORKER_WIDTH)
        .thread_name(|index| format!("online-pa-parity-oracle-{index}"))
        .build()?;
    anyhow::ensure!(
        oracle_pool.current_num_threads() == CPU_WORKER_WIDTH,
        "PB-v2 parity oracle did not receive seven Rayon workers"
    );
    let raw = deterministic_matrix(ROWS, cols);
    let cpu_probabilities =
        lease.scope(|| pb_v2_cpu_oracle(&raw, &original_labels, 1.0, 1, &oracle_pool))?;
    drop(raw);

    install_device_policy("auto");
    let frame = deterministic_frame(ROWS, cols)?;
    let mut model = OnlinePassiveAggressiveExpert::new(1.0, 1);
    model.fit(&frame, &original_labels, &lease)?;
    let gpu_probabilities = model.predict_proba(&frame, &lease)?;
    assert_probability_rows_close(&cpu_probabilities, &gpu_probabilities, 2.0e-5);
    let runtime = model.predict_runtime(&frame.row_window(0, 1)?, &lease)?;
    assert_runtime_is_proven_cuda_without_degradation(&runtime);

    let artifact_dir = TempArtifactDir::create("online-pa-million-row-parity")?;
    model.save(artifact_dir.path())?;
    let artifact: Value =
        serde_json::from_slice(&std::fs::read(artifact_dir.path().join("model.json"))?)?;
    let pipeline = artifact
        .pointer("/cuda_training_evidence/full_pipeline")
        .context("million-row parity full CUDA receipt")?;
    let chunks_per_epoch = ROWS.div_ceil(1_024) as u64;
    assert_eq!(
        pipeline["training_row_chunk_count_per_epoch"],
        chunks_per_epoch
    );
    assert_eq!(pipeline["training_launch_count"], chunks_per_epoch);
    assert_eq!(pipeline["training_interchunk_device_to_host_bytes"], 0);
    Ok(json!({
        "schema": "neoethos.online_pa.million_row_pb_v2_parity.v1",
        "rows": ROWS,
        "cols": cols,
        "epochs": 1,
        "worker_width": CPU_WORKER_WIDTH,
        "cpu_checksum": weighted_probability_checksum(&cpu_probabilities),
        "gpu_checksum": weighted_probability_checksum(&gpu_probabilities),
        "effective_device_policy": pipeline["effective_device_policy"],
        "device_uuid": pipeline["device_identity"]["uuid"],
        "training_launch_count": pipeline["training_launch_count"],
    }))
}

struct SerialRtxRunner {
    active_child: Option<String>,
    next_sequence: u64,
}

impl SerialRtxRunner {
    fn new() -> Self {
        Self {
            active_child: None,
            next_sequence: 0,
        }
    }

    fn no_concurrent_gpu_children(&self) -> bool {
        self.active_child.is_none()
    }

    fn run(
        &mut self,
        role: &str,
        receipt_path: &Path,
        timeout: Duration,
        requires_gpu: bool,
    ) -> Result<Value> {
        anyhow::ensure!(
            self.active_child.is_none(),
            "refusing to start `{role}` while {:?} is active",
            self.active_child
        );
        let sequence = self.next_sequence;
        self.active_child = Some(role.to_string());
        let spawned = Command::new(std::env::current_exe()?)
            .arg("--exact")
            .arg("online_pa_rtx_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(RTX_CHILD_ROLE_ENV, role)
            .env(RTX_CHILD_RECEIPT_ENV, receipt_path)
            .env(RTX_CHILD_SEQUENCE_ENV, sequence.to_string())
            .spawn()
            .with_context(|| format!("spawn isolated RTX child `{role}`"));
        let child = match spawned {
            Ok(child) => child,
            Err(error) => {
                self.active_child = None;
                return Err(error);
            }
        };
        let wait_result = wait_child_with_timeout(child, timeout, role);
        self.active_child = None;
        wait_result?;

        let receipt: Value = serde_json::from_slice(
            &std::fs::read(receipt_path).with_context(|| format!("read `{role}` child receipt"))?,
        )?;
        anyhow::ensure!(receipt["status"] == "ok", "`{role}` did not report success");
        anyhow::ensure!(receipt["role"] == role, "`{role}` receipt role drift");
        anyhow::ensure!(
            receipt["sequence"] == sequence,
            "`{role}` receipt sequence drift"
        );
        let payload = receipt["payload"].clone();
        if requires_gpu {
            anyhow::ensure!(
                payload["effective_device_policy"]
                    .as_str()
                    .is_some_and(|policy| policy.starts_with("gpu:")),
                "`{role}` did not bind an effective gpu:N"
            );
            anyhow::ensure!(
                payload["device_uuid"].as_array().is_some_and(|uuid| {
                    uuid.len() == 16
                        && uuid
                            .iter()
                            .any(|byte| byte.as_u64().unwrap_or_default() != 0)
                }),
                "`{role}` omitted its physical CUDA UUID"
            );
        }
        self.next_sequence += 1;
        Ok(payload)
    }
}

#[test]
fn online_pa_rtx_child() -> Result<()> {
    let Some(role) = std::env::var_os(RTX_CHILD_ROLE_ENV) else {
        return Ok(());
    };
    let role = role.to_string_lossy().into_owned();
    let sequence = std::env::var(RTX_CHILD_SEQUENCE_ENV)
        .context("RTX child sequence")?
        .parse::<u64>()?;
    let payload = match role.as_str() {
        "lifecycle-auto" => exercise_full_cuda_lifecycle("auto")?,
        "lifecycle-gpu-0" => exercise_full_cuda_lifecycle("gpu:0")?,
        "parity-64" => run_million_row_parity_role(64)?,
        "parity-128" => run_million_row_parity_role(128)?,
        "benchmark-pb-v2-cpu7" | "benchmark-gpu" | "benchmark-legacy-cpu" => {
            run_benchmark_role(&role)?
        }
        "max-epoch-64" => run_max_epoch_role(64)?,
        "max-epoch-128" => run_max_epoch_role(128)?,
        other => bail!("unknown isolated RTX child role `{other}`"),
    };
    let receipt = json!({
        "status": "ok",
        "role": role,
        "sequence": sequence,
        "payload": payload,
    });
    let path = std::env::var_os(RTX_CHILD_RECEIPT_ENV).context("RTX child receipt path")?;
    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}

#[test]
#[ignore = "requires a real RTX device; runs every online PA GPU gate serially in fresh timeout-bounded child processes"]
fn online_pa_rtx_validation_serial_orchestrator() -> Result<()> {
    let receipts = TempArtifactDir::create("online-pa-r4-serial-rtx")?;
    let mut runner = SerialRtxRunner::new();
    let mut run = |role: &str, timeout: Duration, requires_gpu: bool| {
        let receipt_path = receipts.path().join(format!("{role}.json"));
        runner.run(role, &receipt_path, timeout, requires_gpu)
    };

    let lifecycle_auto = run("lifecycle-auto", LIFECYCLE_WALL_TIMEOUT, true)?;
    assert_eq!(lifecycle_auto["requested_device_policy"], "auto");
    let lifecycle_gpu_0 = run("lifecycle-gpu-0", LIFECYCLE_WALL_TIMEOUT, true)?;
    assert_eq!(lifecycle_gpu_0["requested_device_policy"], "gpu:0");
    assert_eq!(lifecycle_gpu_0["effective_device_policy"], "gpu:0");

    for (role, cols) in [("parity-64", 64), ("parity-128", 128)] {
        let parity = run(role, PARITY_WALL_TIMEOUT, true)?;
        assert_eq!(parity["rows"], 1_000_000);
        assert_eq!(parity["cols"], cols);
        assert_eq!(parity["worker_width"], CPU_WORKER_WIDTH);
    }

    let cpu = run("benchmark-pb-v2-cpu7", BENCH_WALL_TIMEOUT, false)?;
    let gpu = run("benchmark-gpu", BENCH_WALL_TIMEOUT, true)?;
    let legacy = run("benchmark-legacy-cpu", BENCH_WALL_TIMEOUT, false)?;
    validate_benchmark_gate(&cpu, &gpu, &legacy)?;

    for (role, cols) in [("max-epoch-64", 64), ("max-epoch-128", 128)] {
        let receipt = run(role, MAX_EPOCH_WALL_TIMEOUT, true)?;
        assert_eq!(receipt["rows"], 1_000_000);
        assert_eq!(receipt["cols"], cols);
        assert_eq!(receipt["epochs"], ONLINE_PA_SEARCH_MAX_EPOCHS);
        assert_eq!(receipt["worker_width"], CPU_WORKER_WIDTH);
    }

    drop(run);
    let ratchet = json!({
        "no_concurrent_gpu_children": runner.no_concurrent_gpu_children(),
        "completed_child_count": runner.next_sequence,
    });
    assert_eq!(ratchet["no_concurrent_gpu_children"], true);
    assert_eq!(ratchet["completed_child_count"], 9);
    Ok(())
}
