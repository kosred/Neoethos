use std::process::Command;
use std::sync::{Arc, Barrier};

use neoethos_core::Settings;
use neoethos_core::config::FeatureCubeMode;
use neoethos_data::core::hpc_ta::{IndicatorComputePolicy, set_indicator_compute_policy};
use neoethos_search::{
    BacktestRuntimeOverrides, CanonicalNativeDiscoveryRequestErrorV1 as Error,
    CanonicalNativeGenerationZeroRuntimeAuthorityV1, CanonicalNativeRuntimeInstallReceiptV1,
    EvaluationBackend, GeneticSearchRuntimeOverrides, SeenSignatureMemoryRuntimeOverrides,
    StrategyEvaluationRuntimeOverrides, current_evaluation_backend,
    install_and_seal_canonical_native_runtime_authority_v1, install_backtest_runtime_overrides,
    install_evaluation_backend, install_genetic_search_runtime_overrides,
    install_search_runtime_overrides_from_settings,
    install_seen_signature_memory_runtime_overrides, install_smc_search_config_from_settings,
    install_stop_target_runtime_overrides_from_settings,
    install_strategy_evaluation_runtime_overrides,
};

const CASE_ENV: &str = "NEOETHOS_CHUNK1B_AUTHORITY_CASE";

fn settings() -> Settings {
    let mut value = Settings::default();
    value.models.seen_signature_runtime.max_entries = 3_000_000;
    value
}

fn assert_runtime_error(result: Result<CanonicalNativeRuntimeInstallReceiptV1, Error>) {
    assert!(matches!(result, Err(Error::RuntimeAuthority(_))));
}

#[test]
fn authority_case_driver() {
    let Ok(case) = std::env::var(CASE_ENV) else {
        return;
    };
    let base = settings();
    match case.as_str() {
        "data" => neoethos_data::install_data_runtime_overrides(
            !base.models.data_runtime.normalize_features,
        ),
        "cube" => neoethos_data::install_feature_cube_policy(FeatureCubeMode::Disk),
        "genetic" => {
            let mut value = GeneticSearchRuntimeOverrides::default();
            value.novelty_weight = 0.25;
            install_genetic_search_runtime_overrides(value).unwrap();
        }
        "strategy" => {
            let mut value = StrategyEvaluationRuntimeOverrides::default();
            value.cost_profile.symbol = Some("GBPUSD".to_owned());
            install_strategy_evaluation_runtime_overrides(value).unwrap();
        }
        "backtest" => {
            let mut value = BacktestRuntimeOverrides::default();
            value.initial_equity += 1.0;
            install_backtest_runtime_overrides(value).unwrap();
        }
        "smc" => {
            let mut wrong = base.clone();
            wrong.models.smc_search_runtime.p_ob = 0.91;
            install_smc_search_config_from_settings(&wrong);
        }
        "stop" => {
            let mut wrong = base.clone();
            wrong.risk.atr_stop_multiplier = 2.5;
            install_stop_target_runtime_overrides_from_settings(&wrong);
        }
        "gene" => {
            let mut wrong = base.clone();
            wrong.models.gene_stop_bounds.atr_scaled = !base.models.gene_stop_bounds.atr_scaled;
            install_search_runtime_overrides_from_settings(&wrong);
        }
        "seen" => {
            let mut value = SeenSignatureMemoryRuntimeOverrides::from_settings(&base);
            value.flush_every += 1;
            install_seen_signature_memory_runtime_overrides(value).unwrap();
        }
        "seen-zero" => {
            let zero = Settings::default();
            let receipt = install_and_seal_canonical_native_runtime_authority_v1(&zero).unwrap();
            assert_eq!(
                receipt.identity_sha256(),
                install_and_seal_canonical_native_runtime_authority_v1(&zero)
                    .unwrap()
                    .identity_sha256()
            );
            return;
        }
        "indicator" => set_indicator_compute_policy(IndicatorComputePolicy::CpuOnly).unwrap(),
        "backend" => {
            let receipt = install_and_seal_canonical_native_runtime_authority_v1(&base).unwrap();
            install_evaluation_backend(EvaluationBackend::CPU_CANONICAL).unwrap();
            assert_runtime_error(install_and_seal_canonical_native_runtime_authority_v1(
                &base,
            ));
            assert!(!receipt.identity_sha256().is_empty());
            return;
        }
        "map-order" => {
            let mut first = base.clone();
            first
                .models
                .prop_search_max_rows_by_tf
                .insert("M1".to_owned(), 10);
            first
                .models
                .prop_search_max_rows_by_tf
                .insert("H1".to_owned(), 20);
            let mut second = base.clone();
            second
                .models
                .prop_search_max_rows_by_tf
                .insert("H1".to_owned(), 20);
            second
                .models
                .prop_search_max_rows_by_tf
                .insert("M1".to_owned(), 10);
            let receipt = install_and_seal_canonical_native_runtime_authority_v1(&first).unwrap();
            let repeated = install_and_seal_canonical_native_runtime_authority_v1(&second).unwrap();
            assert_eq!(receipt.identity_sha256(), repeated.identity_sha256());
            let mut conflict = first.clone();
            conflict.models.prop_search_device = "cpu".to_owned();
            assert_runtime_error(install_and_seal_canonical_native_runtime_authority_v1(
                &conflict,
            ));
            assert_eq!(
                current_evaluation_backend(),
                EvaluationBackend::from_settings_and_process_env(&first).unwrap()
            );
            assert_eq!(
                receipt.identity_sha256(),
                install_and_seal_canonical_native_runtime_authority_v1(&first)
                    .unwrap()
                    .identity_sha256()
            );
            return;
        }
        "concurrent" => {
            let mut other = base.clone();
            other.models.prop_search_device = "cpu".to_owned();
            let barrier = Arc::new(Barrier::new(3));
            let launch = |value: Settings, barrier: Arc<Barrier>| {
                std::thread::spawn(move || {
                    barrier.wait();
                    install_and_seal_canonical_native_runtime_authority_v1(&value)
                        .map(|receipt| receipt.identity_sha256().to_owned())
                })
            };
            let left = launch(base.clone(), barrier.clone());
            let right = launch(other.clone(), barrier.clone());
            barrier.wait();
            let (left, right) = (left.join().unwrap(), right.join().unwrap());
            let (identity, winner) = match (left, right) {
                (Ok(identity), Err(Error::RuntimeAuthority(_))) => (identity, base),
                (Err(Error::RuntimeAuthority(_)), Ok(identity)) => (identity, other),
                result => panic!("expected exactly one serialized winner, got {result:?}"),
            };
            assert_eq!(
                current_evaluation_backend(),
                EvaluationBackend::from_settings_and_process_env(&winner).unwrap()
            );
            assert_eq!(
                identity,
                install_and_seal_canonical_native_runtime_authority_v1(&winner)
                    .unwrap()
                    .identity_sha256()
            );
            return;
        }
        unknown => panic!("unknown authority case {unknown}"),
    }
    assert_runtime_error(install_and_seal_canonical_native_runtime_authority_v1(
        &base,
    ));
}

#[test]
fn every_runtime_class_and_install_race_is_checked_in_a_fresh_process() {
    for case in [
        "data",
        "cube",
        "genetic",
        "strategy",
        "backtest",
        "smc",
        "stop",
        "gene",
        "seen",
        "seen-zero",
        "indicator",
        "backend",
        "map-order",
        "concurrent",
    ] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "authority_case_driver", "--nocapture"])
            .env(CASE_ENV, case)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "case {case} failed\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn receipt_and_authority_are_opaque_installer_products() {
    let _ = std::mem::size_of::<CanonicalNativeRuntimeInstallReceiptV1>();
    let _ = std::mem::size_of::<CanonicalNativeGenerationZeroRuntimeAuthorityV1>();
    let source = include_str!("../src/canonical_native_runtime_authority_v1.rs");
    assert_eq!(
        source
            .matches("SeenSignatureMemoryRuntimeOverrides::from_settings")
            .count(),
        1
    );
    assert!(source.contains("install_settings.models.seen_signature_runtime.max_entries"));
    assert!(source.contains("expected.seen_memory.max_entries"));
    assert!(source.contains("invoke_runtime_installers(&install_settings)"));
}
