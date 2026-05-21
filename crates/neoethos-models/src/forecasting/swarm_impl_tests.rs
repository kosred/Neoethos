// TODO(real-data): every dataframe / value vector in this file is
// synthesised (linear ramps, cyclic class labels, hand-written floats).
// Replace every fixture below with a cTrader historical sample for
// the symbol/timeframe the swarm forecaster is intended to run on
// (e.g. EURUSD M5 with 256 closes) so the asserted behaviour reflects
// real broker data shape, including weekend gaps and quote noise.
use super::*;

use polars::prelude::NamedFrom;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn test_artifact_dir(name: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock moved backwards")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "neoethos-swarm-forecasting-{name}-{}-{unique}",
        std::process::id()
    ))
}

#[test]
fn extract_series_rejects_class_like_target_column_as_price_source() {
    let frame = DataFrame::new(vec![
        Series::new(
            "feature".into(),
            (0..64).map(|idx| idx as f64).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "target".into(),
            (0..64)
                .map(|idx| match idx % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                })
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .expect("frame");
    let labels = Series::new(
        "labels".into(),
        (0..64)
            .map(|idx| match idx % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect::<Vec<_>>(),
    );

    let err = extract_series_from_frame(&frame, &labels)
        .expect_err("class-like target column must not be forecast as price");
    assert!(err.to_string().contains("price-like series"));
}

#[test]
fn extract_series_uses_continuous_labels_when_price_column_is_absent() {
    let frame = DataFrame::new(vec![
        Series::new(
            "feature".into(),
            (0..64).map(|idx| idx as f64).collect::<Vec<_>>(),
        )
        .into(),
    ])
    .expect("frame");
    let labels = Series::new(
        "continuous_target".into(),
        (0..64)
            .map(|idx| 1.10_f64 + idx as f64 * 0.0001)
            .collect::<Vec<_>>(),
    );

    let values = extract_series_from_frame(&frame, &labels).expect("continuous label series");
    assert_eq!(values.len(), 64);
    assert!(values[63] > values[0]);
}

#[test]
fn load_preserves_saved_local_fallback_artifacts_without_refitting() {
    let dir = test_artifact_dir("load-preserves-local-artifact");
    fs::create_dir_all(&dir).expect("create temp dir");

    let values = vec![1.0, 1.2, 1.1, 1.4, 1.6, 1.5, 1.7, 1.8];
    let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");
    let last_result = SwarmForecastResult {
        point_forecast: vec![1.9, 2.0, 2.1],
        level_80_lower: vec![1.7, 1.8, 1.9],
        level_80_upper: vec![2.1, 2.2, 2.3],
        diversity_score: 0.12,
        effective_models: 3.0,
        prediction_variance: 0.04,
        models_used: 3,
        runtime_backend_kind: Some(BackendKind::LocalSurrogateFallback),
        runtime_mode: Some(RuntimeMode::Degraded),
        runtime_degraded_reason: Some(RuntimeDegradedReason::new(
            "swarm_local_fallback_active",
            "swarm_local_fallback_active",
        )),
    };
    let candidate_reports = vec![SwarmCandidateReport {
        name: "persistence".to_string(),
        model_type: "MLP".to_string(),
        source: "local".to_string(),
        weight: 1.0,
        prediction_length: 3,
        prediction_mean: 2.0,
        prediction_std: 0.2,
        mae: 0.1,
        mse: 0.01,
        mape: 5.0,
        smape: 4.0,
        coverage: 0.8,
        bias: 0.0,
        directional_accuracy: 0.75,
        regime_fit: 0.65,
        stability_score: 0.90,
    }];
    let training_report = SwarmTrainingReport {
        training_rows: values.len(),
        validation_windows: 0,
        fitted_horizon: 3,
        best_candidate: Some("persistence".to_string()),
        aggregate_mae: 0.1,
        aggregate_smape: 4.0,
        aggregate_directional_accuracy: 0.75,
        aggregate_coverage: 0.8,
        diversity_score: 0.12,
        regime_bias: 0.2,
        updated_at_unix_ms: Some(456),
    };

    let artifact = SwarmForecasterArtifact {
        config: SwarmForecastConfig::default(),
        runtime_mode: SwarmRuntimeMode::LocalFallback,
        runtime_degraded_reason: Some("swarm_local_fallback_active".to_string()),
        fitted: true,
        values: values.clone(),
        timestamps: timestamps.clone(),
        unique_id: "test-series".to_string(),
        snapshot: Some(snapshot.clone()),
        last_result: Some(last_result.clone()),
        last_horizon: Some(3),
        candidate_reports: candidate_reports.clone(),
        updated_at_unix_ms: Some(123),
        training_report: Some(training_report.clone()),
    };
    let payload = serde_json::to_vec_pretty(&artifact).expect("serialize artifact");
    fs::write(dir.join(SWARM_ARTIFACT_FILE_NAME), payload).expect("write artifact");

    let saved_candidate_reports_len = candidate_reports.len();
    let mut forecaster = SwarmForecaster::new(256.0);
    forecaster.load(&dir).expect("load local fallback artifact");

    assert_eq!(forecaster.runtime_mode, SwarmRuntimeMode::LocalFallback);
    assert!(forecaster.fitted);
    assert_eq!(forecaster.values, values);
    assert_eq!(forecaster.timestamps, timestamps);
    assert_eq!(forecaster.unique_id, "test-series");
    assert_eq!(forecaster.last_horizon, Some(3));
    assert_eq!(forecaster.updated_at_unix_ms, Some(123));
    let loaded_result = forecaster.last_result.expect("last result after load");
    assert_eq!(loaded_result.point_forecast, last_result.point_forecast);
    assert_eq!(loaded_result.level_80_lower, last_result.level_80_lower);
    assert_eq!(loaded_result.level_80_upper, last_result.level_80_upper);
    assert_eq!(loaded_result.diversity_score, last_result.diversity_score);
    assert_eq!(loaded_result.effective_models, last_result.effective_models);
    assert_eq!(
        loaded_result.prediction_variance,
        last_result.prediction_variance
    );
    assert_eq!(loaded_result.models_used, last_result.models_used);
    assert_eq!(
        loaded_result.runtime_backend_kind,
        Some(BackendKind::LocalSurrogateFallback)
    );
    assert_eq!(loaded_result.runtime_mode, Some(RuntimeMode::Degraded));
    assert_eq!(
        loaded_result
            .runtime_degraded_reason
            .as_ref()
            .map(|reason| reason.code.as_str()),
        Some("swarm_local_fallback_active")
    );

    assert_eq!(forecaster.candidate_reports.len(), candidate_reports.len());
    let loaded_report = &forecaster.candidate_reports[0];
    let expected_report = &candidate_reports[0];
    assert_eq!(loaded_report.name, expected_report.name);
    assert_eq!(loaded_report.model_type, expected_report.model_type);
    assert_eq!(loaded_report.source, expected_report.source);
    assert_eq!(loaded_report.weight, expected_report.weight);
    assert_eq!(
        loaded_report.prediction_length,
        expected_report.prediction_length
    );
    assert_eq!(
        loaded_report.prediction_mean,
        expected_report.prediction_mean
    );
    assert_eq!(loaded_report.prediction_std, expected_report.prediction_std);
    assert_eq!(loaded_report.mae, expected_report.mae);
    assert_eq!(loaded_report.mse, expected_report.mse);
    assert_eq!(loaded_report.mape, expected_report.mape);
    assert_eq!(loaded_report.smape, expected_report.smape);
    assert_eq!(loaded_report.coverage, expected_report.coverage);
    assert_eq!(loaded_report.bias, expected_report.bias);
    assert_eq!(
        loaded_report.directional_accuracy,
        expected_report.directional_accuracy
    );
    assert_eq!(loaded_report.regime_fit, expected_report.regime_fit);
    assert_eq!(
        loaded_report.stability_score,
        expected_report.stability_score
    );

    let loaded_snapshot = forecaster.snapshot.expect("snapshot after load");
    assert_eq!(loaded_snapshot.last_value, snapshot.last_value);
    assert_eq!(loaded_snapshot.rolling_mean, snapshot.rolling_mean);
    assert_eq!(loaded_snapshot.drift_slope, snapshot.drift_slope);
    assert_eq!(loaded_snapshot.volatility, snapshot.volatility);
    assert_eq!(loaded_snapshot.has_trend, snapshot.has_trend);
    assert_eq!(loaded_snapshot.has_seasonality, snapshot.has_seasonality);
    assert_eq!(loaded_snapshot.seasonal_periods, snapshot.seasonal_periods);
    assert_eq!(loaded_snapshot.short_mean, snapshot.short_mean);
    assert_eq!(loaded_snapshot.medium_mean, snapshot.medium_mean);
    assert_eq!(loaded_snapshot.long_mean, snapshot.long_mean);
    assert_eq!(loaded_snapshot.recent_return, snapshot.recent_return);
    assert_eq!(loaded_snapshot.trend_strength, snapshot.trend_strength);
    assert_eq!(
        loaded_snapshot.mean_reversion_strength,
        snapshot.mean_reversion_strength
    );
    assert_eq!(loaded_snapshot.volatility_ratio, snapshot.volatility_ratio);

    let loaded_training_report = forecaster
        .training_report
        .expect("training report after load");
    assert_eq!(
        loaded_training_report.training_rows,
        training_report.training_rows
    );
    assert_eq!(
        loaded_training_report.validation_windows,
        training_report.validation_windows
    );
    assert_eq!(
        loaded_training_report.fitted_horizon,
        training_report.fitted_horizon
    );
    assert_eq!(
        loaded_training_report.best_candidate,
        training_report.best_candidate
    );
    assert_eq!(
        loaded_training_report.aggregate_mae,
        training_report.aggregate_mae
    );
    assert_eq!(
        loaded_training_report.aggregate_smape,
        training_report.aggregate_smape
    );
    assert_eq!(
        loaded_training_report.aggregate_directional_accuracy,
        training_report.aggregate_directional_accuracy
    );
    assert_eq!(
        loaded_training_report.aggregate_coverage,
        training_report.aggregate_coverage
    );
    assert_eq!(
        loaded_training_report.diversity_score,
        training_report.diversity_score
    );
    assert_eq!(
        loaded_training_report.regime_bias,
        training_report.regime_bias
    );
    assert_eq!(
        loaded_training_report.updated_at_unix_ms,
        training_report.updated_at_unix_ms
    );

    // F-MODELS9-007 fix: per-field equality is necessary but not
    // sufficient. Aggregate invariants of the loaded artifact must
    // also hold, otherwise a deserialization bug that re-normalizes
    // (or drops) entries could pass the per-field asserts above.
    if !forecaster.candidate_reports.is_empty() {
        let total_weight: f32 = forecaster.candidate_reports.iter().map(|r| r.weight).sum();
        // Weights need only be POSITIVE and finite after load; not
        // necessarily summing to 1.0 because the saved test fixture
        // doesn't pre-normalize. The contract we enforce: count
        // preserved + all weights finite + ordering preserved.
        assert!(
            total_weight.is_finite() && total_weight > 0.0,
            "loaded candidate weights must be finite + positive (got {total_weight})"
        );
        assert_eq!(
            forecaster.candidate_reports.len(),
            saved_candidate_reports_len,
            "loaded report count must match what was saved"
        );
        for report in &forecaster.candidate_reports {
            assert!(
                report.weight.is_finite() && report.weight >= 0.0,
                "candidate '{}' must have a finite non-negative weight after load (got {})",
                report.name,
                report.weight
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn load_rebuilds_stale_fitted_artifact_diagnostics() {
    let dir = test_artifact_dir("rebuilds-stale-artifact");
    fs::create_dir_all(&dir).expect("create temp dir");

    let values = (0..96)
        .map(|idx| 1.0 + idx as f32 * 0.01 + ((idx % 7) as f32 - 3.0) * 0.02)
        .collect::<Vec<_>>();
    let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");

    let artifact = SwarmForecasterArtifact {
        config: SwarmForecastConfig {
            horizon: 12,
            ..SwarmForecastConfig::default()
        },
        runtime_mode: SwarmRuntimeMode::LocalFallback,
        runtime_degraded_reason: Some("stale_external_state".to_string()),
        fitted: true,
        values: values.clone(),
        timestamps: timestamps.clone(),
        unique_id: "stale-series".to_string(),
        snapshot: Some(snapshot),
        last_result: Some(SwarmForecastResult {
            point_forecast: vec![1.0, 1.1],
            level_80_lower: vec![0.9, 1.0],
            level_80_upper: vec![1.1, 1.2],
            diversity_score: 0.0,
            effective_models: 1.0,
            prediction_variance: 0.0,
            models_used: 1,
            runtime_backend_kind: Some(BackendKind::LocalSurrogateFallback),
            runtime_mode: Some(RuntimeMode::Degraded),
            runtime_degraded_reason: Some(RuntimeDegradedReason::new(
                "stale_external_state",
                "stale_external_state",
            )),
        }),
        last_horizon: Some(12),
        candidate_reports: vec![SwarmCandidateReport {
            name: "stale".to_string(),
            model_type: "local".to_string(),
            source: "stale".to_string(),
            weight: 1.0,
            prediction_length: 2,
            prediction_mean: 1.05,
            prediction_std: 0.1,
            mae: 0.2,
            mse: 0.04,
            mape: 10.0,
            smape: 8.0,
            coverage: 0.5,
            bias: 0.0,
            directional_accuracy: 0.5,
            regime_fit: 0.5,
            stability_score: 0.5,
        }],
        updated_at_unix_ms: None,
        training_report: Some(SwarmTrainingReport {
            training_rows: values.len(),
            validation_windows: 0,
            fitted_horizon: 6,
            best_candidate: Some("stale".to_string()),
            aggregate_mae: 0.2,
            aggregate_smape: 8.0,
            aggregate_directional_accuracy: 0.5,
            aggregate_coverage: 0.5,
            diversity_score: 0.0,
            regime_bias: 0.0,
            updated_at_unix_ms: None,
        }),
    };
    let payload = serde_json::to_vec_pretty(&artifact).expect("serialize artifact");
    fs::write(dir.join(SWARM_ARTIFACT_FILE_NAME), payload).expect("write artifact");

    let mut forecaster = SwarmForecaster::new(256.0);
    forecaster.load(&dir).expect("load stale artifact");

    assert!(forecaster.fitted);
    assert_eq!(forecaster.last_horizon, Some(12));
    assert!(forecaster.last_result.is_some());
    assert!(!forecaster.candidate_reports.is_empty());
    assert!(forecaster.training_report.is_some());
    assert!(
        forecaster
            .last_result
            .as_ref()
            .is_some_and(|result| result_is_valid(result, 12))
    );
    assert!(candidate_reports_are_valid(
        &forecaster.candidate_reports,
        12
    ));
    assert!(
        forecaster
            .training_report
            .as_ref()
            .is_some_and(|report| report.fitted_horizon == 12)
    );
    assert_eq!(
        forecaster.runtime_degraded_reason.as_deref(),
        Some("swarm_local_fallback_active")
    );

    let _ = fs::remove_dir_all(&dir);
}

// F-MODELS9-013: the rebuild test above only exercises the happy path
// where the persisted horizon (12) is feasible for the saved series
// (96 values). It never checks what happens when an artifact is saved
// with more horizon than data. A 12-step forecast genuinely cannot be
// produced from 8 observations: validation_windows will be empty
// (build_validation_windows needs >= 16 + 4 rows for horizon 12) so
// there is no out-of-sample signal at all, and moving_average_long
// (window=16) silently collapses onto the same 8-value mean as the
// medium window, eliminating diversity. The correct load-path
// behaviour is to either:
//   (a) bail with an error about horizon-vs-history feasibility, or
//   (b) clamp the persisted horizon down to a value supportable by the
//       available history (e.g. values.len() / 2) and surface a
//       degradation reason naming the clamp.
//
// F-MODELS9-013-impl SHIPPED 2026-05-15 (commit 3dce122) — the loader
// at `swarm_impl.rs:251-270` now bails with an explicit
// "horizon exceeds half of available history" message when a persisted
// artifact's horizon is unsupportable on its history. This test runs
// (no `#[ignore]`) and exercises Case (a) of the contract below;
// Case (b) is kept in the assertion ladder so a future migration to
// "clamp-and-degrade" semantics can swap one branch for the other
// without churning the test.
#[test]
fn load_rejects_or_downgrades_artifact_with_incompatible_horizon() {
    let dir = test_artifact_dir("incompatible-horizon-artifact");
    fs::create_dir_all(&dir).expect("create temp dir");

    // 8 observations is the absolute minimum the local snapshot
    // builder accepts. Pair it with a persisted horizon of 12, which
    // is unsupportable on this history (validation requires at least
    // ~20 rows for horizon 12).
    let values = vec![1.0_f32; 8];
    let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");

    let artifact = SwarmForecasterArtifact {
        config: SwarmForecastConfig {
            horizon: 12,
            ..SwarmForecastConfig::default()
        },
        runtime_mode: SwarmRuntimeMode::LocalFallback,
        runtime_degraded_reason: None,
        fitted: true,
        values: values.clone(),
        timestamps: timestamps.clone(),
        unique_id: "incompatible-horizon".to_string(),
        snapshot: Some(snapshot),
        last_result: None,
        last_horizon: Some(12),
        candidate_reports: Vec::new(),
        updated_at_unix_ms: None,
        training_report: None,
    };
    let payload = serde_json::to_vec_pretty(&artifact).expect("serialize artifact");
    fs::write(dir.join(SWARM_ARTIFACT_FILE_NAME), payload).expect("write artifact");

    let mut forecaster = SwarmForecaster::new(256.0);
    let outcome = forecaster.load(&dir);

    match outcome {
        Err(err) => {
            // Case (a): bail. The error must explicitly name the
            // horizon-vs-history conflict rather than fail at some
            // unrelated invariant.
            let message = err.to_string().to_ascii_lowercase();
            assert!(
                message.contains("horizon")
                    && (message.contains("history")
                        || message.contains("observations")
                        || message.contains("values")),
                "expected horizon/history error, got: {err}"
            );
        }
        Ok(()) => {
            // Case (b): downgrade. The loaded forecaster must NOT
            // claim to forecast 12 steps from 8 observations. The
            // effective horizon must be reduced to something the
            // history supports (at most values.len() / 2 == 4) and
            // the degradation must be surfaced.
            let effective_horizon = forecaster
                .last_horizon
                .expect("last_horizon set after load");
            assert!(
                effective_horizon <= forecaster.values.len() / 2,
                "load must clamp horizon (got {effective_horizon}) to at most values.len()/2 ({}) when persisted horizon exceeds history",
                forecaster.values.len() / 2
            );
            if let Some(result) = forecaster.last_result.as_ref() {
                assert_eq!(
                    result.point_forecast.len(),
                    effective_horizon,
                    "rebuilt forecast length must match the (clamped) effective horizon"
                );
            }
            let reason = forecaster
                .runtime_degraded_reason
                .as_deref()
                .expect("downgrade must surface a degradation reason");
            assert!(
                reason.contains("horizon"),
                "degradation reason must name the horizon clamp, got: {reason}"
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(not(feature = "swarm-forecasting"))]
#[test]
fn load_downgrades_persisted_external_runtime_when_feature_is_disabled() {
    let dir = test_artifact_dir("load-downgrades-external-runtime");
    fs::create_dir_all(&dir).expect("create temp dir");

    let values = vec![1.0, 1.2, 1.1, 1.4, 1.6, 1.5, 1.7, 1.8];
    let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");
    let last_result = SwarmForecastResult {
        point_forecast: vec![1.9, 2.0, 2.1],
        level_80_lower: vec![1.7, 1.8, 1.9],
        level_80_upper: vec![2.1, 2.2, 2.3],
        diversity_score: 0.12,
        effective_models: 3.0,
        prediction_variance: 0.04,
        models_used: 3,
        runtime_backend_kind: Some(BackendKind::ExternalRuntime),
        runtime_mode: Some(RuntimeMode::Canonical),
        runtime_degraded_reason: None,
    };
    let candidate_reports = vec![SwarmCandidateReport {
        name: "persistence".to_string(),
        model_type: "MLP".to_string(),
        source: "external".to_string(),
        weight: 1.0,
        prediction_length: 3,
        prediction_mean: 2.0,
        prediction_std: 0.2,
        mae: 0.1,
        mse: 0.01,
        mape: 5.0,
        smape: 4.0,
        coverage: 0.8,
        bias: 0.0,
        directional_accuracy: 0.75,
        regime_fit: 0.65,
        stability_score: 0.90,
    }];
    let training_report = SwarmTrainingReport {
        training_rows: values.len(),
        validation_windows: 1,
        fitted_horizon: 3,
        best_candidate: Some("persistence".to_string()),
        aggregate_mae: 0.1,
        aggregate_smape: 4.0,
        aggregate_directional_accuracy: 0.75,
        aggregate_coverage: 0.8,
        diversity_score: 0.12,
        regime_bias: 0.2,
        updated_at_unix_ms: Some(456),
    };

    let artifact = SwarmForecasterArtifact {
        config: SwarmForecastConfig::default(),
        runtime_mode: SwarmRuntimeMode::ExternalSwarm,
        runtime_degraded_reason: None,
        fitted: true,
        values: values.clone(),
        timestamps: timestamps.clone(),
        unique_id: "test-series".to_string(),
        snapshot: Some(snapshot),
        last_result: Some(last_result),
        last_horizon: Some(3),
        candidate_reports,
        updated_at_unix_ms: Some(123),
        training_report: Some(training_report),
    };
    let payload = serde_json::to_vec_pretty(&artifact).expect("serialize artifact");
    fs::write(dir.join(SWARM_ARTIFACT_FILE_NAME), payload).expect("write artifact");

    let mut forecaster = SwarmForecaster::new(256.0);
    forecaster.load(&dir).expect("load external artifact");

    assert_eq!(forecaster.runtime_mode, SwarmRuntimeMode::LocalFallback);
    assert_eq!(
        forecaster.runtime_degraded_reason.as_deref(),
        Some("swarm_forecasting_feature_disabled")
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "swarm-forecasting")]
#[test]
fn select_external_or_fallback_result_marks_local_fallback_when_external_result_is_invalid() {
    let snapshot = SwarmForecastSnapshot {
        last_value: 1.0,
        rolling_mean: 1.0,
        drift_slope: 0.02,
        volatility: 0.05,
        has_trend: true,
        has_seasonality: false,
        seasonal_periods: vec![],
        short_mean: 1.01,
        medium_mean: 1.0,
        long_mean: 0.99,
        recent_return: 0.01,
        trend_strength: 0.55,
        mean_reversion_strength: 0.20,
        volatility_ratio: 1.05,
    };
    let candidates = vec![
        ("fast".to_string(), vec![1.01, 1.03]),
        ("slow".to_string(), vec![0.99, 1.01]),
    ];
    let reports = vec![
        SwarmCandidateReport {
            name: "fast".to_string(),
            model_type: "MLP".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.6,
            prediction_length: 2,
            prediction_mean: 1.02,
            prediction_std: 0.01,
            mae: 0.05,
            mse: 0.004,
            mape: 2.0,
            smape: 2.0,
            coverage: 0.8,
            bias: 0.01,
            directional_accuracy: 0.65,
            regime_fit: 0.60,
            stability_score: 0.62,
        },
        SwarmCandidateReport {
            name: "slow".to_string(),
            model_type: "Transformer".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.4,
            prediction_length: 2,
            prediction_mean: 1.0,
            prediction_std: 0.01,
            mae: 0.06,
            mse: 0.005,
            mape: 2.5,
            smape: 2.5,
            coverage: 0.82,
            bias: -0.01,
            directional_accuracy: 0.61,
            regime_fit: 0.58,
            stability_score: 0.59,
        },
    ];
    let weight_map = build_candidate_weight_map(&reports);
    let invalid_external = SwarmForecastResult {
        point_forecast: vec![f32::NAN, 1.02],
        level_80_lower: vec![0.9, 0.95],
        level_80_upper: vec![1.1, 1.08],
        diversity_score: 0.15,
        effective_models: 2.0,
        prediction_variance: 0.04,
        models_used: 2,
        runtime_backend_kind: Some(BackendKind::ExternalRuntime),
        runtime_mode: Some(RuntimeMode::Canonical),
        runtime_degraded_reason: None,
    };

    let (final_result, runtime_mode, degraded_reason) = select_external_or_fallback_result(
        &snapshot,
        &candidates,
        &weight_map,
        &reports,
        SwarmEnsembleStrategy::BayesianModelAveraging,
        2,
        invalid_external,
    );

    assert_eq!(runtime_mode, SwarmRuntimeMode::LocalFallback);
    assert_eq!(
        degraded_reason.as_deref(),
        Some("external_swarm_result_invalid")
    );
    assert!(result_is_valid(&final_result, 2));
    assert_eq!(
        final_result.runtime_backend_kind,
        Some(BackendKind::LocalSurrogateFallback)
    );
    assert_eq!(final_result.runtime_mode, Some(RuntimeMode::Degraded));
    assert_eq!(
        final_result
            .runtime_degraded_reason
            .as_ref()
            .map(|reason| reason.code.as_str()),
        Some("external_swarm_result_invalid")
    );
}

#[test]
fn save_strips_stale_forecast_state_from_unfitted_artifacts() {
    let dir = test_artifact_dir("save-strips-unfitted-state");
    fs::create_dir_all(&dir).expect("create temp dir");

    let values = vec![1.0, 1.2, 1.1, 1.4, 1.6, 1.5, 1.7, 1.8];
    let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");
    let last_result = SwarmForecastResult {
        point_forecast: vec![1.9, 2.0, 2.1],
        level_80_lower: vec![1.7, 1.8, 1.9],
        level_80_upper: vec![2.1, 2.2, 2.3],
        diversity_score: 0.12,
        effective_models: 3.0,
        prediction_variance: 0.04,
        models_used: 3,
        runtime_backend_kind: Some(BackendKind::ExternalRuntime),
        runtime_mode: Some(RuntimeMode::Canonical),
        runtime_degraded_reason: None,
    };

    let mut forecaster = SwarmForecaster::new(256.0);
    forecaster.runtime_mode = SwarmRuntimeMode::ExternalSwarm;
    forecaster.runtime_degraded_reason = Some("stale_external_state".to_string());
    forecaster.values = values.clone();
    forecaster.timestamps = timestamps.clone();
    forecaster.unique_id = "test-series".to_string();
    forecaster.snapshot = Some(snapshot);
    forecaster.last_result = Some(last_result);
    forecaster.last_horizon = Some(3);
    forecaster.candidate_reports = vec![SwarmCandidateReport {
        name: "stale".to_string(),
        model_type: "MLP".to_string(),
        source: "local".to_string(),
        weight: 1.0,
        prediction_length: 3,
        prediction_mean: 2.0,
        prediction_std: 0.2,
        mae: 0.1,
        mse: 0.01,
        mape: 5.0,
        smape: 4.0,
        coverage: 0.8,
        bias: 0.0,
        directional_accuracy: 0.75,
        regime_fit: 0.65,
        stability_score: 0.90,
    }];
    forecaster.updated_at_unix_ms = Some(123);
    forecaster.training_report = Some(SwarmTrainingReport {
        training_rows: values.len(),
        validation_windows: 1,
        fitted_horizon: 3,
        best_candidate: Some("stale".to_string()),
        aggregate_mae: 0.1,
        aggregate_smape: 4.0,
        aggregate_directional_accuracy: 0.75,
        aggregate_coverage: 0.8,
        diversity_score: 0.12,
        regime_bias: 0.2,
        updated_at_unix_ms: Some(456),
    });

    forecaster.save(&dir).expect("save should succeed");
    let payload = fs::read(dir.join(SWARM_ARTIFACT_FILE_NAME)).expect("read saved swarm artifact");
    let artifact: SwarmForecasterArtifact =
        serde_json::from_slice(&payload).expect("deserialize saved artifact");

    assert!(!artifact.fitted);
    assert_eq!(artifact.runtime_mode, SwarmRuntimeMode::LocalFallback);
    assert!(artifact.runtime_degraded_reason.is_none());
    assert!(artifact.snapshot.is_none());
    assert!(artifact.last_result.is_none());
    assert!(artifact.last_horizon.is_none());
    assert!(artifact.candidate_reports.is_empty());
    assert!(artifact.updated_at_unix_ms.is_none());
    assert!(artifact.training_report.is_none());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn learned_validation_weights_favor_lower_loss_candidates() {
    let mut losses = HashMap::new();
    losses.insert("best".to_string(), 0.12);
    losses.insert("mid".to_string(), 0.48);
    losses.insert("worst".to_string(), 1.44);

    let mut support = HashMap::new();
    support.insert("best".to_string(), 3.0);
    support.insert("mid".to_string(), 3.0);
    support.insert("worst".to_string(), 3.0);

    let weights = derive_validation_candidate_weights(&losses, &support);

    let best = weights.get("best").copied().expect("best weight");
    let mid = weights.get("mid").copied().expect("mid weight");
    let worst = weights.get("worst").copied().expect("worst weight");

    assert!(
        best > mid,
        "best candidate should receive more weight than mid"
    );
    assert!(
        mid > worst,
        "mid candidate should receive more weight than worst"
    );

    let total = weights.values().copied().sum::<f32>();
    assert!((total - 1.0).abs() < 1e-5, "weights should normalize to 1");
}

#[test]
fn learned_validation_weights_edge_cases() {
    // F-MODELS9-008 fix: the ordering-only assertion above doesn't
    // cover the boundary cases. Pin the contract for:
    //   (a) all losses equal → all weights equal,
    //   (b) one candidate has zero (perfect) loss → it dominates,
    //   (c) all losses are NaN → the function falls back to uniform
    //       (or returns Err — assert whichever the implementation
    //       chose so the contract is locked in).

    // (a) Equal losses → equal weights.
    let mut equal_losses = HashMap::new();
    equal_losses.insert("a".to_string(), 0.5);
    equal_losses.insert("b".to_string(), 0.5);
    equal_losses.insert("c".to_string(), 0.5);
    let mut equal_support = HashMap::new();
    equal_support.insert("a".to_string(), 3.0);
    equal_support.insert("b".to_string(), 3.0);
    equal_support.insert("c".to_string(), 3.0);
    let equal_weights = derive_validation_candidate_weights(&equal_losses, &equal_support);
    let wa = equal_weights.get("a").copied().expect("a");
    let wb = equal_weights.get("b").copied().expect("b");
    let wc = equal_weights.get("c").copied().expect("c");
    assert!(
        (wa - wb).abs() < 1e-5 && (wb - wc).abs() < 1e-5,
        "equal losses must produce equal weights (got a={wa}, b={wb}, c={wc})"
    );
    let equal_total = equal_weights.values().copied().sum::<f32>();
    assert!(
        (equal_total - 1.0).abs() < 1e-5,
        "equal-loss weights must still normalize to 1"
    );

    // (b) Near-zero loss for one candidate → it should dominate.
    let mut perfect_losses = HashMap::new();
    perfect_losses.insert("perfect".to_string(), 1e-6_f32);
    perfect_losses.insert("bad".to_string(), 2.0);
    let mut perfect_support = HashMap::new();
    perfect_support.insert("perfect".to_string(), 5.0);
    perfect_support.insert("bad".to_string(), 5.0);
    let perfect_weights = derive_validation_candidate_weights(&perfect_losses, &perfect_support);
    let w_perfect = perfect_weights.get("perfect").copied().expect("perfect");
    let w_bad = perfect_weights.get("bad").copied().expect("bad");
    assert!(
        w_perfect > w_bad,
        "perfect candidate must outweigh bad one (got perfect={w_perfect}, bad={w_bad})"
    );

    // (c) Document NaN-handling: if the impl returns Err on all-NaN,
    // assert that; otherwise check the fallback distribution.
    let mut nan_losses = HashMap::new();
    nan_losses.insert("x".to_string(), f32::NAN);
    nan_losses.insert("y".to_string(), f32::NAN);
    let mut nan_support = HashMap::new();
    nan_support.insert("x".to_string(), 3.0);
    nan_support.insert("y".to_string(), 3.0);
    let nan_weights = derive_validation_candidate_weights(&nan_losses, &nan_support);
    // Implementation choice: must produce finite, normalized weights
    // (no NaN in the result map). If the contract changes to return
    // Err, update this assertion accordingly.
    for (name, weight) in nan_weights.iter() {
        assert!(
            weight.is_finite(),
            "weight for '{name}' must be finite even when all losses are NaN (got {weight})"
        );
    }
    let nan_total = nan_weights.values().copied().sum::<f32>();
    assert!(
        (nan_total - 1.0).abs() < 1e-5 || nan_weights.is_empty(),
        "NaN-loss fallback must normalize or return empty (got total {nan_total})"
    );
}

#[test]
fn normalize_candidate_weights_preserves_learned_validation_ordering() {
    let mut reports = vec![
        SwarmCandidateReport {
            name: "best".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.62,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.55,
            mse: 0.45,
            mape: 4.0,
            smape: 4.0,
            coverage: 0.72,
            bias: 0.02,
            directional_accuracy: 0.66,
            regime_fit: 0.62,
            stability_score: 0.61,
        },
        SwarmCandidateReport {
            name: "mid".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.28,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.20,
            mse: 0.12,
            mape: 2.0,
            smape: 2.0,
            coverage: 0.84,
            bias: 0.01,
            directional_accuracy: 0.77,
            regime_fit: 0.76,
            stability_score: 0.79,
        },
        SwarmCandidateReport {
            name: "worst".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.10,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.08,
            mse: 0.05,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.92,
            bias: 0.0,
            directional_accuracy: 0.88,
            regime_fit: 0.89,
            stability_score: 0.91,
        },
    ];

    normalize_candidate_weights(&mut reports);

    assert!(
        reports[0].weight > reports[1].weight,
        "existing learned weights should keep best ahead of mid"
    );
    assert!(
        reports[1].weight > reports[2].weight,
        "existing learned weights should keep mid ahead of worst"
    );
    let total = reports.iter().map(|report| report.weight).sum::<f32>();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "normalized learned weights should still sum to 1"
    );
}

#[test]
fn validation_weight_blend_ratio_scales_with_support() {
    let low_support = HashMap::from([("a".to_string(), 1.0_f32), ("b".to_string(), 1.0_f32)]);
    let high_support = HashMap::from([("a".to_string(), 24.0_f32), ("b".to_string(), 24.0_f32)]);

    let low_ratio = validation_weight_blend_ratio(&low_support, 2);
    let high_ratio = validation_weight_blend_ratio(&high_support, 2);

    assert!(
        low_ratio < 0.8,
        "low support should preserve heuristic influence"
    );
    assert!(
        high_ratio > 0.95,
        "strong validation support should rely almost entirely on learned weights"
    );
    assert!(high_ratio > low_ratio);
}

#[test]
fn training_report_matches_requires_exact_validation_window_count() {
    let reports = vec![SwarmCandidateReport {
        name: "best".to_string(),
        model_type: "local".to_string(),
        source: "rolling_validation".to_string(),
        weight: 1.0,
        prediction_length: 8,
        prediction_mean: 1.0,
        prediction_std: 0.1,
        mae: 0.1,
        mse: 0.1,
        mape: 1.0,
        smape: 1.0,
        coverage: 0.9,
        bias: 0.0,
        directional_accuracy: 0.8,
        regime_fit: 0.8,
        stability_score: 0.8,
    }];
    let report = SwarmTrainingReport {
        training_rows: 128,
        validation_windows: 2,
        fitted_horizon: 8,
        best_candidate: Some("best".to_string()),
        aggregate_mae: 0.1,
        aggregate_smape: 1.0,
        aggregate_directional_accuracy: 0.8,
        aggregate_coverage: 0.9,
        diversity_score: 0.2,
        regime_bias: 0.1,
        updated_at_unix_ms: Some(1),
    };

    assert!(
        !training_report_matches(&report, &reports, 128, 3, 8),
        "stale validation-window counts should invalidate persisted swarm reports"
    );
}

#[test]
fn rolling_validation_reports_use_forecast_horizon_not_validation_window_length() {
    let train_values = (0..72)
        .map(|idx| 1.0 + idx as f32 * 0.01 + ((idx % 5) as f32 - 2.0) * 0.002)
        .collect::<Vec<_>>();
    let train_timestamps = (0..train_values.len())
        .map(|idx| idx as f64)
        .collect::<Vec<_>>();
    let actuals = vec![1.72, 1.73, 1.74, 1.75];
    let windows = vec![(train_values, train_timestamps, actuals)];

    let reports = build_weighted_reports_local(&windows, 12).expect("rolling reports");

    assert!(!reports.is_empty());
    assert!(reports.iter().all(|report| report.prediction_length == 12));
    assert!(candidate_reports_are_valid(&reports, 12));
    assert!(!candidate_reports_are_valid(&reports, 4));
}

#[test]
fn candidate_reports_are_valid_rejects_duplicate_names() {
    let reports = vec![
        SwarmCandidateReport {
            name: "dup".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.6,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
        SwarmCandidateReport {
            name: "dup".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.4,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
    ];

    assert!(!candidate_reports_are_valid(&reports, 8));
}

#[test]
fn candidate_reports_are_valid_rejects_non_positive_weights() {
    let reports = vec![SwarmCandidateReport {
        name: "a".to_string(),
        model_type: "local".to_string(),
        source: "rolling_validation".to_string(),
        weight: 0.0,
        prediction_length: 8,
        prediction_mean: 1.0,
        prediction_std: 0.1,
        mae: 0.1,
        mse: 0.1,
        mape: 1.0,
        smape: 1.0,
        coverage: 0.9,
        bias: 0.0,
        directional_accuracy: 0.9,
        regime_fit: 0.9,
        stability_score: 0.9,
    }];

    assert!(!candidate_reports_are_valid(&reports, 8));
}

#[test]
fn prune_validation_candidates_keeps_top_supported_reports() {
    let mut reports = vec![
        SwarmCandidateReport {
            name: "a".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.42,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
        SwarmCandidateReport {
            name: "b".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.28,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
        SwarmCandidateReport {
            name: "c".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.14,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
        SwarmCandidateReport {
            name: "d".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.09,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
        SwarmCandidateReport {
            name: "e".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.04,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
    ];
    let support = HashMap::from([
        ("a".to_string(), 8.0_f32),
        ("b".to_string(), 8.0_f32),
        ("c".to_string(), 8.0_f32),
        ("d".to_string(), 8.0_f32),
        ("e".to_string(), 8.0_f32),
    ]);

    prune_validation_candidates(&mut reports, &support);

    assert!(reports.len() < 5);
    assert!(reports.iter().any(|report| report.name == "a"));
    assert!(reports.iter().any(|report| report.name == "b"));
    let total = reports.iter().map(|report| report.weight).sum::<f32>();
    assert!((total - 1.0).abs() < 1e-5);
}

#[test]
fn select_active_candidates_uses_positive_weight_reports() {
    let candidates = vec![
        ("a".to_string(), vec![1.0, 1.1]),
        ("b".to_string(), vec![1.0, 1.2]),
        ("c".to_string(), vec![1.0, 0.9]),
    ];
    let reports = vec![
        SwarmCandidateReport {
            name: "a".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.7,
            prediction_length: 2,
            prediction_mean: 1.05,
            prediction_std: 0.05,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
        SwarmCandidateReport {
            name: "b".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.3,
            prediction_length: 2,
            prediction_mean: 1.1,
            prediction_std: 0.05,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
        SwarmCandidateReport {
            name: "c".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.0,
            prediction_length: 2,
            prediction_mean: 0.95,
            prediction_std: 0.05,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
    ];

    let active = select_active_candidates(&candidates, &reports);
    assert_eq!(active.len(), 2);
    assert!(active.iter().all(|(name, _)| name == "a" || name == "b"));
}

#[test]
fn calibrated_interval_spread_widens_when_validation_coverage_is_weak() {
    let snapshot = SwarmForecastSnapshot {
        last_value: 1.0,
        rolling_mean: 1.0,
        drift_slope: 0.01,
        volatility: 0.03,
        has_trend: true,
        has_seasonality: false,
        seasonal_periods: vec![],
        short_mean: 1.02,
        medium_mean: 1.01,
        long_mean: 0.99,
        recent_return: 0.01,
        trend_strength: 0.7,
        mean_reversion_strength: 0.2,
        volatility_ratio: 1.0,
    };
    let step_values = vec![1.00, 1.02, 1.03, 1.01];
    let strong = [SwarmCandidateReport {
        name: "a".to_string(),
        model_type: "local".to_string(),
        source: "rolling_validation".to_string(),
        weight: 1.0,
        prediction_length: 4,
        prediction_mean: 1.015,
        prediction_std: 0.01,
        mae: 0.01,
        mse: 0.0002,
        mape: 1.0,
        smape: 1.0,
        coverage: 0.88,
        bias: 0.002,
        directional_accuracy: 0.9,
        regime_fit: 0.9,
        stability_score: 0.9,
    }];
    let weak = [SwarmCandidateReport {
        coverage: 0.45,
        mae: 0.03,
        prediction_std: 0.02,
        bias: 0.01,
        ..strong[0].clone()
    }];
    let strong_refs = strong.iter().collect::<Vec<_>>();
    let weak_refs = weak.iter().collect::<Vec<_>>();

    let strong_spread = calibrated_interval_spread(1.015, &step_values, &strong_refs, &snapshot);
    let weak_spread = calibrated_interval_spread(1.015, &step_values, &weak_refs, &snapshot);

    assert!(weak_spread > strong_spread);
}

#[test]
fn active_report_refs_follow_active_candidates() {
    let reports = vec![
        SwarmCandidateReport {
            name: "a".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.6,
            prediction_length: 2,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
        SwarmCandidateReport {
            name: "b".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.4,
            prediction_length: 2,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        },
    ];
    let candidates = vec![("b".to_string(), vec![1.0, 1.1])];

    let active = active_report_refs(&reports, &candidates);

    assert_eq!(active.len(), 1);
    assert_eq!(active[0].name, "b");
}

#[test]
fn prune_local_candidates_deduplicates_near_identical_forecasts() {
    let snapshot = SwarmForecastSnapshot {
        last_value: 1.0,
        rolling_mean: 1.0,
        drift_slope: 0.01,
        volatility: 0.05,
        has_trend: true,
        has_seasonality: false,
        seasonal_periods: vec![],
        short_mean: 1.02,
        medium_mean: 1.01,
        long_mean: 0.99,
        recent_return: 0.01,
        trend_strength: 0.7,
        mean_reversion_strength: 0.2,
        volatility_ratio: 1.0,
    };
    let candidates = vec![
        ("drift".to_string(), vec![1.01, 1.02, 1.03, 1.04]),
        (
            "momentum_blend".to_string(),
            vec![1.0105, 1.0205, 1.0305, 1.0405],
        ),
        ("mean_reversion".to_string(), vec![1.0, 0.995, 0.99, 0.985]),
        ("persistence".to_string(), vec![1.0, 1.0, 1.0, 1.0]),
    ];

    let pruned = prune_local_candidates(&snapshot, candidates);

    assert!(pruned.len() < 4);
    assert!(pruned.iter().any(|(name, _)| name == "drift"));
    assert!(!pruned.iter().any(|(name, _)| name == "momentum_blend"));
}

#[test]
fn candidate_preselection_score_prefers_regime_aligned_family() {
    let trend_snapshot = SwarmForecastSnapshot {
        last_value: 1.0,
        rolling_mean: 0.99,
        drift_slope: 0.02,
        volatility: 0.04,
        has_trend: true,
        has_seasonality: false,
        seasonal_periods: vec![],
        short_mean: 1.03,
        medium_mean: 1.01,
        long_mean: 0.98,
        recent_return: 0.01,
        trend_strength: 0.8,
        mean_reversion_strength: 0.15,
        volatility_ratio: 1.0,
    };
    let drift = vec![1.01, 1.03, 1.05, 1.07];
    let mean_reversion = vec![0.995, 0.992, 0.99, 0.988];

    let drift_score = candidate_preselection_score("drift", &drift, &trend_snapshot);
    let mean_reversion_score =
        candidate_preselection_score("mean_reversion", &mean_reversion, &trend_snapshot);

    assert!(drift_score > mean_reversion_score);
}

#[test]
fn candidate_forecasts_local_reduce_mean_reversion_noise_in_trend_regime() {
    let snapshot = SwarmForecastSnapshot {
        last_value: 1.0,
        rolling_mean: 0.99,
        drift_slope: 0.025,
        volatility: 0.04,
        has_trend: true,
        has_seasonality: false,
        seasonal_periods: vec![],
        short_mean: 1.04,
        medium_mean: 1.02,
        long_mean: 0.98,
        recent_return: 0.012,
        trend_strength: 0.82,
        mean_reversion_strength: 0.18,
        volatility_ratio: 1.0,
    };
    let values = (0..64)
        .map(|idx| 1.0 + idx as f32 * 0.01)
        .collect::<Vec<_>>();

    let candidates = candidate_forecasts_local(&values, &snapshot, 8);

    assert!(candidates.iter().any(|(name, _)| name == "drift"));
    assert!(candidates.iter().any(|(name, _)| name == "damped_drift"));
    assert!(candidates.iter().any(|(name, _)| name == "momentum_blend"));
    assert!(!candidates.iter().any(|(name, _)| name == "mean_reversion"));
    assert!(!candidates.iter().any(|(name, _)| name == "regime_anchor"));
}

#[cfg(feature = "swarm-forecasting")]
#[test]
fn candidate_predictions_reduce_trend_noise_in_mean_reversion_regime() {
    let snapshot = SwarmForecastSnapshot {
        last_value: 1.0,
        rolling_mean: 1.01,
        drift_slope: -0.002,
        volatility: 0.05,
        has_trend: false,
        has_seasonality: false,
        seasonal_periods: vec![],
        short_mean: 1.0,
        medium_mean: 1.01,
        long_mean: 1.02,
        recent_return: -0.001,
        trend_strength: 0.25,
        mean_reversion_strength: 0.76,
        volatility_ratio: 1.1,
    };
    let values = (0..64)
        .map(|idx| 1.05 - idx as f32 * 0.0015)
        .collect::<Vec<_>>();

    let candidates = candidate_predictions(&values, &snapshot, 8);

    assert!(
        candidates
            .iter()
            .any(|(name, _, _)| name == "mean_reversion")
    );
    assert!(
        candidates
            .iter()
            .any(|(name, _, _)| name == "regime_anchor")
    );
    assert!(
        !candidates
            .iter()
            .any(|(name, _, _)| name == "momentum_blend")
    );
}

#[cfg(feature = "swarm-forecasting")]
#[test]
fn sanitize_downgrades_stale_external_runtime_to_local_fallback_reason() {
    let values = (0..64)
        .map(|idx| 1.0 + idx as f32 * 0.01 + ((idx % 5) as f32 - 2.0) * 0.002)
        .collect::<Vec<_>>();
    let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");

    let artifact = SwarmForecasterArtifact {
        config: SwarmForecastConfig::default(),
        runtime_mode: SwarmRuntimeMode::ExternalSwarm,
        runtime_degraded_reason: None,
        fitted: true,
        values,
        timestamps,
        unique_id: "repair-external".to_string(),
        snapshot: Some(snapshot),
        last_result: None,
        last_horizon: Some(6),
        candidate_reports: Vec::new(),
        updated_at_unix_ms: Some(1),
        training_report: None,
    };

    let repaired = sanitize_forecaster_artifact(artifact).expect("sanitize");
    assert_eq!(repaired.runtime_mode, SwarmRuntimeMode::LocalFallback);
    assert_eq!(
        repaired.runtime_degraded_reason.as_deref(),
        Some("external_swarm_result_rebuilt_from_local_consensus")
    );
    assert!(repaired.last_result.is_some());
    assert!(!repaired.candidate_reports.is_empty());
}
