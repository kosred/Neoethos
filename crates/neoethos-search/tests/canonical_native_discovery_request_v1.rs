#![cfg(all(feature = "gpu-cuda", target_os = "linux"))]

use std::path::Path;
use std::process::Command;

use neoethos_core::Settings;
use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::core::features::{FeatureBuildOptions, FeatureProfile};
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1,
    CanonicalOhlcvPublishRequest, CanonicalTimeframe, CanonicalVolumeRef,
    SelectedDatasetGenerationV1, publish_canonical_ohlcv_generation,
};
use neoethos_search::data_selection::CanonicalSearchInput;
use neoethos_search::{
    CanonicalNativeCostBandStatusV1, CanonicalNativeDiscoveryRequestErrorV1 as Error,
    CanonicalNativeExecutionScopeV1, CanonicalNativeGenerationZeroOverridesV1,
    CanonicalResearchContractArtifactRefV1, CanonicalTrendbarResearchCostAssumptionsV2,
    CanonicalTrendbarResearchExecutionContractV3, EvaluationBackend,
    MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1, MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1,
    MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1,
    install_and_seal_canonical_native_runtime_authority_v1, install_evaluation_backend,
    resolve_canonical_native_discovery_request_v1, set_migration_enabled,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const CASE_ENV: &str = "NEOETHOS_CHUNK1B_REQUEST_CASE";
const ARTIFACT: &str = "research/contracts/chunk1b.json";

fn compatible_settings(root: &Path) -> Settings {
    let mut settings = Settings::default();
    settings.system.data_dir = root.to_owned();
    settings.system.symbol = "EURUSD".to_owned();
    settings.system.account_currency = "USD".to_owned();
    settings.system.base_timeframe = "H4".to_owned();
    settings.system.higher_timeframes = vec!["D1".to_owned()];
    settings.models.prop_search_population = 100;
    settings.models.prop_search_population_auto = false;
    settings.models.prop_search_generations = 50;
    settings.models.prop_search_max_indicators = 12;
    settings.models.prop_search_max_rows = 0;
    settings.models.prop_search_max_rows_by_tf.clear();
    settings.models.prop_search_min_payoff_ratio = 0.0;
    settings.models.prop_search_device = "auto".to_owned();
    settings.models.discovery_runtime.prefilter_top_k = 0;
    settings.models.discovery_runtime.min_history_years = 0;
    settings.models.discovery_runtime.adaptive_thresholds = false;
    settings.models.discovery_ledger.enabled = false;
    settings.models.gene_stop_bounds.atr_scaled = false;
    settings
}

fn publish(root: &Path, expected: Option<&str>, tag: &str) -> SelectedDatasetGenerationV1 {
    let identity = CanonicalDatasetIdentity::external(
        "neoethos-chunk1b-request",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .unwrap();
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.search.chunk1b-request-fixture.v1",
        tag.as_bytes().to_vec(),
    )
    .unwrap();
    let bars = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
    let published = publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity: &identity,
        expected_generation: expected,
        provenance: &provenance,
        ohlcv: &bars,
        volume: CanonicalVolumeRef::Float64(bars.volume.as_deref().unwrap()),
        rows_per_chunk: 64,
    })
    .unwrap();
    SelectedDatasetGenerationV1::from_manifest(published.manifest()).unwrap()
}

fn write_contract(root: &Path, selected: &SelectedDatasetGenerationV1) -> String {
    let series =
        CanonicalDatasetSeriesReceiptV1::new(selected.clone(), vec![selected.clone()]).unwrap();
    let input = CanonicalSearchInput::from_exact_series_receipt(
        root,
        &series,
        CanonicalTimeframe::M1,
        &FeatureBuildOptions::default(),
    )
    .unwrap();
    let receipt = input.receipt().unwrap();
    let assumption_sha = format!("{:x}", Sha256::digest(b"chunk1b-financial-values"));
    let contract = CanonicalTrendbarResearchExecutionContractV3::new(
        receipt,
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "neoethos.test.chunk1b-financial-values.v1",
            assumption_source_sha256: &assumption_sha,
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.2,
            slippage_pips_per_fill_assumption: 0.1,
            commission_account_per_lot_per_fill_assumption: 3.5,
            swap_long_pips_per_day: -0.2,
            swap_short_pips_per_day: -0.1,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .unwrap();
    let bytes = serde_json::to_vec(&contract).unwrap();
    let path = root.join(ARTIFACT);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, &bytes).unwrap();
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn request_case_driver() {
    let Ok(case) = std::env::var(CASE_ENV) else {
        return;
    };
    let root = TempDir::new().unwrap();
    let first = publish(root.path(), None, "generation-one");
    let artifact_sha = write_contract(root.path(), &first);
    let mut settings = compatible_settings(root.path());
    match case.as_str() {
        "session" => {
            settings.risk.backtest_spread_pips_asian = Some(1.0);
            settings.risk.backtest_spread_pips_overlap = Some(1.1);
            settings.risk.backtest_spread_pips_late_ny = Some(1.2);
        }
        "session-malformed" => settings.risk.backtest_spread_pips_asian = Some(1.0),
        "adaptive" => settings.models.discovery_runtime.adaptive_thresholds = true,
        "atr" => settings.models.gene_stop_bounds.atr_scaled = true,
        "history" => settings.models.discovery_runtime.min_history_years = 1,
        "ledger" => settings.models.discovery_ledger.enabled = true,
        "row" => settings.models.prop_search_max_rows = 50,
        "prefilter" => settings.models.discovery_runtime.prefilter_top_k = 1,
        "symbol" => settings.system.symbol = "GBPUSD".to_owned(),
        "account" => settings.system.account_currency = "EUR".to_owned(),
        "legacy-zero" => settings.models.prop_search_generations = 0,
        "ledger-cache-string-cap" => {
            settings.models.discovery_ledger.cache_dir =
                "x".repeat(MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1 + 1);
        }
        "row-key-string-cap" => {
            settings
                .models
                .prop_search_max_rows_by_tf
                .insert("x".repeat(MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1 + 1), 0);
        }
        "row-map-source-cap" => {
            settings.models.prop_search_max_rows_by_tf.extend(
                (0..=MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1)
                    .map(|index| (format!("tf-{index}"), 0)),
            );
        }
        _ => {}
    }
    let install = install_and_seal_canonical_native_runtime_authority_v1(&settings).unwrap();
    if case == "stale" {
        let _ = publish(root.path(), Some(first.generation_id()), "generation-two");
    }
    if case == "migration-resolve" {
        set_migration_enabled(true);
    }
    let reference =
        CanonicalResearchContractArtifactRefV1::checked_new(ARTIFACT, artifact_sha).unwrap();
    let overrides =
        CanonicalNativeGenerationZeroOverridesV1::checked_new(Some(128), Some(true), Some(0))
            .unwrap();
    let result =
        resolve_canonical_native_discovery_request_v1(&settings, &install, reference, overrides);
    match case.as_str() {
        "session" | "session-malformed" | "adaptive" | "atr" | "history" | "ledger" | "row"
        | "prefilter" => assert!(matches!(
            result,
            Err(Error::UnsupportedGenerationZeroPolicy { .. })
        )),
        "symbol" | "account" => {
            assert!(matches!(result, Err(Error::ContractSettingsMismatch(_))))
        }
        "stale" => {
            let mut native_preflight_called = false;
            if result.is_ok() {
                native_preflight_called = true;
            }
            assert!(matches!(
                result,
                Err(Error::ExactDatasetGenerationConflict(_))
            ));
            assert!(!native_preflight_called);
        }
        "migration-resolve" => assert!(matches!(result, Err(Error::MigrationEnabled))),
        "ledger-cache-string-cap" | "row-key-string-cap" => assert!(matches!(
            result,
            Err(Error::RequestLimitExceeded {
                limit: "string_bytes_cap"
            })
        )),
        "row-map-source-cap" => assert!(matches!(
            result,
            Err(Error::RequestLimitExceeded {
                limit: "source_count_cap"
            })
        )),
        "migration-preflight" => {
            let request = result.unwrap();
            set_migration_enabled(true);
            assert!(matches!(
                request.revalidate_before_native_preflight_v1(&settings),
                Err(Error::MigrationEnabled)
            ));
        }
        "backend-preflight" => {
            let request = result.unwrap();
            install_evaluation_backend(EvaluationBackend::CPU_CANONICAL).unwrap();
            assert!(matches!(
                request.revalidate_before_native_preflight_v1(&settings),
                Err(Error::RuntimeAuthority(_))
            ));
        }
        "legacy-zero" => {
            let request = result.unwrap();
            assert_eq!(
                request.scope().raw_legacy_generations_unused_full_search(),
                0
            );
            assert_eq!(
                request
                    .scope()
                    .clamped_legacy_generations_unused_full_search(),
                1
            );
        }
        "happy" => {
            let request = result.unwrap();
            assert_eq!(request.config().population, 128);
            assert!(request.config().population_auto);
            assert_eq!(request.config().max_indicators, 0);
            assert_eq!(request.config().timeframe_label, "M1");
            assert!(request.config().higher_timeframes.is_empty());
            assert_eq!(request.feature_profile(), FeatureProfile::Standard);
            assert_eq!(
                request.scope().execution_scope(),
                CanonicalNativeExecutionScopeV1::GenerationZeroOnly
            );
            assert_eq!(
                serde_json::to_string(&request.scope().execution_scope()).unwrap(),
                "\"generation_zero_only\""
            );
            assert_eq!(
                request.scope().raw_legacy_generations_unused_full_search(),
                50
            );
            assert_eq!(
                request.scope().cost_band_status(),
                CanonicalNativeCostBandStatusV1::UnusedGenerationZero
            );
            assert_eq!(
                serde_json::to_string(&request.scope().cost_band_status()).unwrap(),
                "\"unused_generation_zero\""
            );
            assert_eq!(
                request.scope().cost_band_pips_unused_generation_zero(),
                request.config().cost_band_pips
            );
            assert_eq!(
                request.exact_series().anchor().generation_id(),
                first.generation_id()
            );
            assert_eq!(request.limits().source_count_cap(), 14);
            assert_eq!(
                request.limits().result_bytes_cap(),
                MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1
            );
            assert_eq!(
                request.startup_settings_sha256(),
                install.startup_settings_sha256()
            );
            request
                .revalidate_before_native_preflight_v1(&settings)
                .unwrap();
        }
        unknown => panic!("unknown request case {unknown}"),
    }
}

#[test]
fn request_policy_cases_run_in_fresh_processes() {
    for case in [
        "happy",
        "legacy-zero",
        "session",
        "session-malformed",
        "adaptive",
        "atr",
        "history",
        "ledger",
        "row",
        "prefilter",
        "symbol",
        "account",
        "stale",
        "migration-resolve",
        "migration-preflight",
        "backend-preflight",
        "ledger-cache-string-cap",
        "row-key-string-cap",
        "row-map-source-cap",
    ] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "request_case_driver", "--nocapture"])
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
fn override_wire_is_strict_and_bounded_before_artifact_access() {
    assert!(CanonicalNativeGenerationZeroOverridesV1::checked_new(Some(0), None, None).is_err());
    assert!(
        CanonicalNativeGenerationZeroOverridesV1::checked_new(Some(1_000_001), None, None).is_err()
    );
    assert!(
        CanonicalNativeGenerationZeroOverridesV1::checked_new(None, None, Some(4_097)).is_err()
    );
    assert!(
        serde_json::from_str::<CanonicalNativeGenerationZeroOverridesV1>(
            r#"{"population":100,"unknown":true}"#
        )
        .is_err()
    );
}
