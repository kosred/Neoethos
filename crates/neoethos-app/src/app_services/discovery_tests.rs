use super::*;

use crate::app_services::{
    ServiceEvent,
    jobs::{JobKind, JobSnapshot, JobState},
};
use neoethos_core::execution::BudgetedCpuExecutor;
use neoethos_core::execution_budget::{AcquireError, CpuPermitBroker, WorkerLimit};
use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CanonicalOhlcvPublishRequest, CanonicalVolumeRef, Ohlcv,
    publish_canonical_ohlcv_generation,
};
use neoethos_search::Gene;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

fn test_execution(width: usize) -> (CpuPermitBroker, Arc<AppExecutionState>) {
    let width = WorkerLimit::new(width).expect("nonzero test width");
    let broker = CpuPermitBroker::new(width);
    let execution = Arc::new(
        AppExecutionState::new(broker.clone(), width).expect("test admission coordinator starts"),
    );
    (broker, execution)
}

fn unique_test_root(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "neoethos-app-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ))
}

fn sample_search_input_receipt() -> neoethos_search::CanonicalSearchInputReceiptV2 {
    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    neoethos_search::CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &features)
        .expect("canonical search test receipt")
}

fn sample_discovery_authority() -> (
    neoethos_search::CanonicalSearchInputReceiptV2,
    neoethos_search::CanonicalSearchArtifactScopeV2,
) {
    let receipt = sample_search_input_receipt();
    let scope = neoethos_search::CanonicalSearchArtifactScopeV2::for_entire_receipt(
        neoethos_search::CanonicalSearchWindowRoleV1::DiscoveryInput,
        receipt.clone(),
    )
    .expect("canonical full DiscoveryInput test scope");
    (receipt, scope)
}

fn publish_two_bar_fixture(
    root: &Path,
    identity: &CanonicalDatasetIdentity,
    close: f64,
    expected_generation: Option<&str>,
) -> SelectedDatasetGenerationV1 {
    let step_ms = identity
        .timeframe()
        .fixed_duration_ms()
        .expect("test fixture uses a fixed-duration timeframe");
    let seed = 1_700_000_000_000_i64;
    let start_ms = seed - seed.rem_euclid(step_ms);
    let ohlcv = Ohlcv {
        timestamp: Some(vec![start_ms, start_ms + step_ms]),
        open: vec![close, close],
        high: vec![close, close],
        low: vec![close, close],
        close: vec![close, close],
        volume: None,
    };
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.app-exact-selection-test.v1",
        identity.canonical_bytes(),
    )
    .expect("test provenance");
    let published = publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity,
        expected_generation,
        provenance: &provenance,
        ohlcv: &ohlcv,
        volume: CanonicalVolumeRef::Absent,
        rows_per_chunk: 2,
    })
    .expect("publish canonical test fixture");
    SelectedDatasetGenerationV1::from_manifest(published.manifest())
        .expect("selected test generation")
}

fn request_for(base: CanonicalTimeframe, higher_tfs: Vec<String>) -> (PathBuf, DiscoveryRequest) {
    let root = unique_test_root("pinned-request");
    std::fs::create_dir_all(&root).expect("create pinned request root");
    let base_identity = CanonicalDatasetIdentity::external(
        "embedded-ctrader-fixture-unverified",
        neoethos_data::test_fixtures::ctrader_sample_symbol(),
        base,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid exact fixture identity");
    let anchor = publish_two_bar_fixture(&root, &base_identity, 1.125, None);
    for label in &higher_tfs {
        let timeframe = label
            .parse::<CanonicalTimeframe>()
            .expect("canonical test higher timeframe");
        let identity = identity_for_timeframe(&base_identity, timeframe)
            .expect("same-series higher timeframe identity");
        publish_two_bar_fixture(&root, &identity, 1.125, None);
    }
    let pinned =
        pin_discovery_input(&root, anchor, &higher_tfs).expect("pin exact test generations");
    let settings_path = root.join("request-settings.yaml");
    let settings_bytes = serde_json::to_vec(&serde_json::json!({
        "system": {
            "symbol": base_identity.symbol_name(),
            "base_timeframe": base.as_str(),
            "higher_timeframes": higher_tfs,
            "data_dir": root,
            "multi_resolution_prefix_base": true
        }
    }))
    .expect("encode test-owned Settings");
    std::fs::write(&settings_path, settings_bytes).expect("write test-owned Settings");
    let settings_source = Arc::new(
        DiscoverySettingsSource::load(&settings_path).expect("capture exact test Settings source"),
    );
    let request = DiscoveryRequest {
        data_root: root.clone(),
        settings_source,
        pinned_input: Arc::new(pinned),
        config: Some(neoethos_search::DiscoveryConfig {
            evaluation_symbol: base_identity.symbol_name().to_owned(),
            timeframe_label: base.as_str().to_owned(),
            higher_timeframes: higher_tfs.clone(),
            ..neoethos_search::DiscoveryConfig::default()
        }),
        overrides: TypedDiscoveryOverridesV1::default(),
        higher_tfs,
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    (root, request)
}

struct TestRequest {
    root: PathBuf,
    request: Option<DiscoveryRequest>,
}

impl std::ops::Deref for TestRequest {
    type Target = DiscoveryRequest;

    fn deref(&self) -> &Self::Target {
        self.request.as_ref().expect("test request is live")
    }
}

impl std::ops::DerefMut for TestRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.request.as_mut().expect("test request is live")
    }
}

impl Drop for TestRequest {
    fn drop(&mut self) {
        drop(self.request.take());
        std::fs::remove_dir_all(&self.root).expect("remove sample discovery test root");
    }
}

fn sample_request() -> TestRequest {
    let (root, request) = request_for(CanonicalTimeframe::M1, Vec::new());
    TestRequest {
        root,
        request: Some(request),
    }
}

fn publish_broker_research_fixture_series(
    root: &Path,
    identity: &CanonicalDatasetIdentity,
    expected_generation: Option<&str>,
    shift: f64,
) -> SelectedDatasetGenerationV1 {
    use broker_history::bootstrap_writer::{
        BrokerTrendbarStreamRequest, publish_broker_trendbar_chunks,
    };
    use neoethos_data::{CanonicalOhlcvChunk, CanonicalVolumeChunk};
    let from = broker_history::CANONICAL_TRENDBAR_SERIES_FROM_MS_V1;
    let to = 1_767_225_600_000_i64;
    let first = from + 3 * 86_400_000;
    // These are two specified January D1 opens, not a fixed-duration D1
    // assumption. The short synthetic series is never a coverage/backtest proof.
    let (timestamps, closes) = if identity.timeframe() == CanonicalTimeframe::D1 {
        (
            vec![first, first + 86_400_000],
            vec![1.20 + shift, 1.25 + shift],
        )
    } else {
        (
            (0..100).map(|index| first + index * 60_000).collect(),
            (0..100)
                .map(|index| {
                    1.1 + shift + index as f64 * 0.0001 + (index as f64 * 0.3).sin() * 0.0002
                })
                .collect::<Vec<_>>(),
        )
    };
    let rows = closes.len();
    let published = publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        configured_root: root,
        identity,
        expected_generation,
        requested_from_ms: from,
        requested_to_ms: to,
        retrieved_unix_ms: to as u64,
        returned_from_ms: timestamps[0],
        returned_to_ms: *timestamps.last().unwrap(),
        row_count: rows as u64,
        chunks: vec![Ok::<_, anyhow::Error>(CanonicalOhlcvChunk {
            timestamp_ms: timestamps,
            open: closes.clone(),
            high: closes.iter().map(|close| close + 0.01).collect(),
            low: closes.iter().map(|close| close - 0.01).collect(),
            close: closes,
            volume: CanonicalVolumeChunk::Int64(vec![10; rows]),
        })],
    })
    .expect("publish test-owned broker-bound prices");
    SelectedDatasetGenerationV1::from_manifest(published.manifest()).unwrap()
}

fn broker_research_request(with_d1: bool, account_currency: &str) -> TestRequest {
    let root = unique_test_root("staged-research");
    std::fs::create_dir_all(&root).unwrap();
    let identity = CanonicalDatasetIdentity::ctrader(
        neoethos_data::CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        42,
        1,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .unwrap();
    let selected = publish_broker_research_fixture_series(&root, &identity, None, 0.0);
    if with_d1 {
        let d1 = identity_for_timeframe(&identity, CanonicalTimeframe::D1).unwrap();
        publish_broker_research_fixture_series(&root, &d1, None, 0.0);
    }
    let settings_path = root.join("settings.yaml");
    let commission = if account_currency == "EUR" {
        4.5
    } else {
        5.625
    };
    std::fs::write(
        &settings_path,
        serde_json::to_vec(&serde_json::json!({
            "system": {
                "symbol": "GBPUSD", "base_timeframe": "H4", "higher_timeframes": ["D1"],
                "data_dir": root, "account_currency": account_currency,
                "multi_resolution_prefix_base": true
            },
            "models": {
                "prop_search_population": 37, "prop_search_generations": 11,
                "prop_search_val_candidates": 80, "prop_search_portfolio_size": 9
            },
            "risk": {
                "backtest_spread_pips": 1.25, "slippage_pips": 0.5,
                "commission_per_lot": commission, "commission_per_lot_is_per_side": true
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let request = DiscoveryRequest {
        data_root: root.clone(),
        settings_source: Arc::new(DiscoverySettingsSource::load(&settings_path).unwrap()),
        pinned_input: Arc::new(pin_discovery_input(&root, selected, &[]).unwrap()),
        higher_tfs: Vec::new(),
        config: None,
        overrides: TypedDiscoveryOverridesV1::checked_new(
            Some(96),
            Some(TypedDiscoveryGenerationOverrideV1::Floor(16)),
            Some(8),
            None,
            Some(144),
            Some(24),
        )
        .unwrap(),
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    TestRequest {
        root,
        request: Some(request),
    }
}

fn capture_test_symbol_contract(
    prepared: &broker_history::symbol_contract_cli::PreparedExactBrokerSymbolContractCaptureV1,
    account_override: Option<i64>,
) -> Result<broker_history::ExactBrokerSymbolContractReceiptV1> {
    use broker_history::ctrader_messages::{
        CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
        CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
    };
    let selected = prepared.binding();
    let binding = broker_history::ExactBrokerSymbolContractBindingV1::new(
        selected.environment(),
        account_override.unwrap_or(selected.account_id()),
        selected.symbol_id(),
        selected.symbol_name(),
    )?;
    let light = serde_json::to_vec(&serde_json::json!({
        "payloadType": CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
        "clientMsgId": "symbol-contract-light-symbols",
        "payload": {"ctidTraderAccountId": binding.account_id(), "symbol": [{
            "symbolId": binding.symbol_id(), "symbolName": binding.symbol_name()
        }]}
    }))?;
    let full = serde_json::to_vec(&serde_json::json!({
        "payloadType": CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
        "clientMsgId": "symbol-contract-full-symbol",
        "payload": {"ctidTraderAccountId": binding.account_id(), "symbol": [{
            "symbolId": binding.symbol_id(), "pipPosition": 4, "lotSize": 10_000_000,
            "commissionType": 1, "preciseTradingCommissionRate": 4_500_000_000_i64,
            "preciseMinCommission": 0, "minCommission": 0, "swapCalculationType": 0,
            "swapLong": -2.5, "swapShort": 0.75, "pnlConversionFeeRate": 13
        }]}
    }))?;
    broker_history::symbol_contract_cli::publish_validated_broker_symbol_contract_response_v1(
        prepared.output_root(),
        &binding,
        &light,
        &full,
    )
}

#[test]
fn admission_dimensions_do_not_authorize_an_unsealed_financial_config() {
    let request = broker_research_request(true, "USD");
    request.validate().unwrap();
    assert!(request.config.is_none());
    assert!(
        request
            .execution_config()
            .unwrap_err()
            .to_string()
            .contains("has not been sealed")
    );
    assert!(requested_discovery_counters(&request).contains(&("population".to_owned(), 96)));
    assert_eq!(request.settings_source.settings().system.symbol, "GBPUSD");
}

#[test]
fn default_relative_cache_prepares_exact_costs_without_broker_io() {
    const CHILD: &str = "NEOETHOS_DISCOVERY_RELATIVE_CACHE_TEST_CHILD";
    const MARKER: &str = "discovery-cache-test-owned-root";
    const COMPLETED: &str = "discovery-cache-test-completed";
    if std::env::var_os(CHILD).is_some() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            std::fs::read(cwd.join(MARKER)).unwrap(),
            b"owned cache test"
        );
        let request = broker_research_request(true, "USD");
        let source = &request.settings_source;
        let original_bytes = source.exact_bytes.clone();
        let original_hash = source.sha256.clone();
        assert_eq!(source.settings().system.cache_dir, Path::new("cache"));
        assert_ne!(source.source_path.parent().unwrap(), cwd);
        assert_eq!(source.discovery_cache_root(), cwd.join("cache/discovery"));
        assert!(
            !cwd.join("cache").exists(),
            "loading must not create the cache"
        );
        let selection_bytes = serde_json::to_vec(request.pinned_input.receipt()).unwrap();
        let output_root = source
            .discovery_cache_root()
            .join("sources")
            .join(format!("{:x}", Sha256::digest(&selection_bytes)));
        let mut captures = 0;
        for reuse in [false, true] {
            let costs = prepare_discovery_screening_costs_with(
                &request,
                &CancellationFlag::new(),
                &output_root,
                |prepared| {
                    assert_eq!(prepared.output_root(), output_root);
                    assert!(prepared.output_root().is_absolute());
                    if reuse {
                        broker_history::symbol_contract_cli::reopen_cached_research_symbol_contract_v1(prepared)?
                            .context("exact synthetic cache must reopen without authentication")
                    } else {
                        captures += 1;
                        capture_test_symbol_contract(prepared, None)
                    }
                },
            ).expect("actual cost preparation accepts the resolved default cache root");
            assert_eq!(costs.settings_source_sha256, original_hash);
            assert_eq!(costs.pip_value_per_lot, 10.0);
            assert_eq!(
                costs
                    .envelope
                    .commission_account_per_lot_per_fill_assumption,
                5.625
            );
            assert_eq!(costs.envelope.pnl_conversion_fee_rate, 0.0);
        }
        assert_eq!(captures, 1);
        assert_eq!(source.settings().system.cache_dir, Path::new("cache"));
        assert_eq!(source.exact_bytes, original_bytes);
        assert_eq!(source.sha256, original_hash);
        assert_eq!(std::fs::read(&source.source_path).unwrap(), original_bytes);
        assert_eq!(
            DiscoverySettingsSource::load(&source.source_path)
                .unwrap()
                .discovery_cache_root(),
            source.discovery_cache_root()
        );
        for area in ["sources", "research"] {
            assert_eq!(
                source.discovery_cache_root().join(area),
                cwd.join("cache/discovery").join(area)
            );
        }

        // Explicit absolute destinations stay exact; loading a missing cache is valid.
        let absolute_cache = cwd.join("another-cache");
        let mut document: serde_json::Value = serde_json::from_slice(&original_bytes).unwrap();
        document["system"]["cache_dir"] = serde_json::to_value(&absolute_cache).unwrap();
        std::fs::write(&source.source_path, serde_json::to_vec(&document).unwrap()).unwrap();
        let absolute_source = DiscoverySettingsSource::load(&source.source_path).unwrap();
        assert_eq!(
            absolute_source.discovery_cache_root(),
            absolute_cache.join("discovery")
        );
        assert_eq!(absolute_source.settings().system.cache_dir, absolute_cache);
        assert!(!absolute_cache.exists());
        assert_eq!(source.discovery_cache_root(), cwd.join("cache/discovery"));
        document["system"]["cache_dir"] = serde_json::json!("");
        std::fs::write(&source.source_path, serde_json::to_vec(&document).unwrap()).unwrap();
        assert!(
            DiscoverySettingsSource::load(&source.source_path)
                .unwrap_err()
                .to_string()
                .contains("resolve configured Discovery cache root")
        );
        std::fs::write(
            cwd.join(COMPLETED),
            b"cost preparation assertions completed",
        )
        .unwrap();
        return;
    }

    // Isolate the real default relative path; never mutate the parent harness CWD/env.
    // Reuse the existing owned test-root cleanup and same-harness child pattern.
    let root = unique_test_root("relative-cache-child");
    std::fs::create_dir(&root).unwrap();
    let _cleanup = TestRequest {
        root: root.clone(),
        request: None,
    };
    let parent_cwd = std::env::current_dir().unwrap();
    std::fs::write(root.join(MARKER), b"owned cache test").unwrap();
    let stdout = root.join("child.stdout.log");
    let stderr = root.join("child.stderr.log");
    let module = module_path!().split_once("::").unwrap().1;
    let test_name =
        format!("{module}::default_relative_cache_prepares_exact_costs_without_broker_io");
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test_name, "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .current_dir(&root)
        .stdout(std::fs::File::create(&stdout).unwrap())
        .stderr(std::fs::File::create(&stderr).unwrap())
        .spawn()
        .expect("spawn isolated default-cache regression");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("could not poll owned default-cache child: {error}");
            }
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            child.wait().expect("join timed-out default-cache child");
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    };
    let out = std::fs::read(&stdout).unwrap();
    let err = std::fs::read(&stderr).unwrap();
    {
        use std::io::Write;
        let mut output = std::io::stdout().lock();
        writeln!(output, "BEGIN DISCOVERY CACHE CHILD STDOUT").unwrap();
        output.write_all(&out).unwrap();
        writeln!(output, "\nEND DISCOVERY CACHE CHILD STDOUT").unwrap();
        let mut errors = std::io::stderr().lock();
        writeln!(errors, "BEGIN DISCOVERY CACHE CHILD STDERR").unwrap();
        errors.write_all(&err).unwrap();
        writeln!(errors, "\nEND DISCOVERY CACHE CHILD STDERR").unwrap();
    }
    assert!(
        status.is_some_and(|status| status.success()),
        "default-cache child failed or timed out: {status:?}"
    );
    assert_eq!(
        std::fs::read(root.join(COMPLETED)).unwrap(),
        b"cost preparation assertions completed"
    );
    assert_eq!(std::env::current_dir().unwrap(), parent_cwd);
}

#[cfg(not(feature = "gpu-nvidia"))]
#[tokio::test]
async fn desktop_costs_and_real_features_seal_selected_config_without_search_or_broker_io() {
    let (broker, execution) = test_execution(2);
    for account_currency in ["USD", "EUR"] {
        let mut request = broker_research_request(true, account_currency);
        let (tx, _rx) = mpsc::channel(8);
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);
        let lease = admit_discovery_cpu_stage(
            &execution,
            &CancellationFlag::new(),
            &tx,
            &mut snapshot,
            "synthetic cost/feature test",
        )
        .await
        .unwrap();
        spawn_discovery_cpu_stage(
            Arc::clone(&execution),
            lease,
            CancellationFlag::new(),
            move |scope| {
                scope.require_current_pool()?;
                assert_eq!(scope.worker_limit().get(), 2);
                let original_bytes = request.settings_source.exact_bytes.clone();
                let cancel = CancellationFlag::new();
                let mut costs = prepare_discovery_screening_costs_with(
                    &request,
                    &cancel,
                    &request.data_root.join("cost-sources"),
                    |prepared| capture_test_symbol_contract(prepared, None),
                )
                .expect("production cost producer with injected exact synthetic broker replies");
                let (pip_value, per_side_commission) = if account_currency == "EUR" {
                    // Independent oracle: USD 10 / 1.25 and USD 5.625 / 1.25.
                    (8.0, 4.5)
                } else {
                    (10.0, 5.625)
                };
                assert_eq!(costs.pip_value_per_lot, pip_value);
                assert_eq!(
                    costs
                        .envelope
                        .commission_account_per_lot_per_fill_assumption,
                    per_side_commission
                );
                assert_eq!(costs.envelope.source_account_id, 42);
                assert_eq!(costs.envelope.symbol, "EURUSD");
                assert_eq!(
                    costs
                        .envelope
                        .source_components
                        .iter()
                        .find(|part| part.role == "settings")
                        .unwrap()
                        .sha256,
                    request.settings_source.sha256
                );
                let dataset = request
                    .pinned_input
                    .take_pinned_series_v1()
                    .unwrap()
                    .into_cpu_dataset_without_native_adapter_v1()
                    .unwrap();
                let original_addresses = ohlcv_buffer_addresses(&dataset.frames["M1"]);
                let input =
                    prepare_cpu_discovery_features(&request, dataset, &[CanonicalTimeframe::M1])
                        .expect("real 100-row production feature frame");
                assert_eq!(
                    ohlcv_buffer_addresses(input.base_frame().ohlcv()),
                    original_addresses
                );
                assert_eq!(input.features().n_samples(), 100);
                assert!(input.features().n_features() > 0);
                assert!(
                    input
                        .features()
                        .names
                        .iter()
                        .all(|name| name.starts_with("M1_"))
                );
                assert!(
                    request.config.is_none(),
                    "features alone do not resolve financial config"
                );
                let receipt = input.receipt().unwrap();
                let owned_settings_sha256 = costs.settings_source_sha256.clone();
                costs.settings_source_sha256 = "0".repeat(64);
                let error = seal_discovery_research_contract(&request, receipt.clone(), &costs)
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("another pinned series or Settings source")
                );
                costs.settings_source_sha256 = owned_settings_sha256;
                let unrelated = sample_request();
                costs.selected_series = unrelated.pinned_input.receipt().clone();
                let error = seal_discovery_research_contract(&request, receipt.clone(), &costs)
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("another pinned series or Settings source")
                );
                costs.selected_series = request.pinned_input.receipt().clone();
                let contract = seal_discovery_research_contract(&request, receipt.clone(), &costs)
                    .expect("bind actual feature generation and the exact cost sources");
                assert_eq!(contract.input_receipt(), &receipt);
                assert_eq!(format!("{:?}", contract.artifact_class()), "ResearchOnly");
                assert_eq!(
                    format!("{:?}", contract.promotion_eligibility()),
                    "NotPromotionEligible"
                );
                let config = request
                    .resolve_research_config(&contract)
                    .expect("staged selected config");
                assert_eq!(config.evaluation_symbol, "EURUSD");
                assert_eq!(config.evaluation_account_currency, account_currency);
                assert_eq!(config.timeframe_label, "M1");
                assert!(config.higher_timeframes.is_empty());
                assert_eq!(config.evaluation_spread_pips, 2.25); // 1.25 spread + two 0.5 fills
                assert_eq!(
                    config.evaluation_commission_per_trade,
                    2.0 * per_side_commission
                );
                assert_eq!(
                    (
                        config.population,
                        config.generations,
                        config.candidate_count,
                        config.portfolio_size
                    ),
                    (96, 16, 144, 24)
                );
                assert_eq!(config.max_indicators, 8);
                request.config = Some(config.clone());
                assert_eq!(
                    format!("{:?}", request.execution_config().unwrap()),
                    format!("{config:?}")
                );
                assert_eq!(request.settings_source.settings().system.symbol, "GBPUSD");
                assert_eq!(
                    request.settings_source.settings().system.base_timeframe,
                    "H4"
                );
                assert_eq!(request.settings_source.exact_bytes, original_bytes);
                assert_eq!(
                    std::fs::read(&request.settings_source.source_path).unwrap(),
                    original_bytes
                );
                // The explicit research adapter never opens the global historical gate.
                assert!(
                    neoethos_search::DiscoveryConfig::try_from_settings(
                        request.settings_source.settings()
                    )
                    .is_err()
                );

                let replacement = publish_broker_research_fixture_series(
                    &request.data_root,
                    request.dataset_identity(),
                    Some(request.pinned_input.receipt().anchor().generation_id()),
                    0.1,
                );
                request.pinned_input =
                    Arc::new(pin_discovery_input(&request.data_root, replacement, &[]).unwrap());
                let error =
                    seal_discovery_research_contract(&request, receipt, &costs).unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("pinned generation/manifest/bytes")
                );
                assert!(!model_targets_path_for(&request.data_root, "EURUSD", "M1").exists());
                Ok(())
            },
        )
        .await
        .unwrap()
        .expect("cost and real feature production runs in the admitted App pool");
        assert_eq!(broker.snapshot().available_permits, 2);
    }
}

#[test]
fn desktop_cost_preparation_refuses_missing_d1_before_metadata_capture() {
    let request = broker_research_request(false, "USD");
    let error = prepare_discovery_screening_costs_with(
        &request,
        &CancellationFlag::new(),
        &request.data_root.join("cost-sources"),
        |_| panic!("missing cost data must refuse before a broker request"),
    )
    .err()
    .expect("direct D1 acquisition required");
    assert!(
        error
            .to_string()
            .contains("D1 cost basis acquisition required")
    );
    assert!(request.config.is_none());
}

#[test]
fn desktop_cost_preparation_refuses_another_account_and_honors_pre_cancel() {
    let request = broker_research_request(true, "USD");
    let cancel = CancellationFlag::new();
    let root = request.data_root.join("cost-sources");
    cancel.request();
    let error = prepare_discovery_screening_costs_with(&request, &cancel, &root, |_| {
        panic!("a cancelled preparation must not contact a broker")
    })
    .err()
    .expect("cancelled before capture");
    assert!(error.to_string().contains("__DISCOVERY_CANCELLED__"));
    assert!(!root.exists());
    let error = prepare_discovery_screening_costs_with(
        &request,
        &CancellationFlag::new(),
        &root,
        |prepared| capture_test_symbol_contract(prepared, Some(43)),
    )
    .err()
    .expect("wrong-account capture must be rejected");
    assert!(error.to_string().contains("another account or symbol"));
    assert!(request.config.is_none());
    assert!(request.pinned_input.pinned_series.lock().unwrap().is_some());
}

#[test]
fn feature_options_bind_the_exact_in_sample_normalization_rows() {
    let request = sample_request();
    let options = request
        .feature_build_options(100)
        .expect("valid 100-row input");
    assert_eq!(options.normalization_training_rows, Some(0..80));
    assert!(options.drop_columns_without_normalization_training_support);
}

#[test]
fn feature_options_honor_the_captured_base_prefix_setting() {
    let request = sample_request();
    assert!(
        request
            .settings_source
            .settings()
            .system
            .multi_resolution_prefix_base
    );
    let options = request
        .feature_build_options(100)
        .expect("valid 100-row input");
    assert!(
        options.prefix_base_features,
        "the real worker must not replace the saved prefix policy with its default"
    );
}

#[test]
fn feature_options_reject_insufficient_rows_before_feature_computation() {
    let request = sample_request();
    for rows in [0, 1, 2, 79] {
        request
            .feature_build_options(rows)
            .expect_err("invalid IS/holdout split must fail before the costly build");
    }
    assert_eq!(
        request
            .feature_build_options(80)
            .unwrap()
            .normalization_training_rows,
        Some(0..64)
    );
}

#[test]
fn original_settings_source_survives_selected_run_edits_and_later_file_changes() {
    let request = sample_request();
    let original = request.settings_source.settings().clone();
    let original_bytes = request.settings_source.exact_bytes.clone();
    let original_hash = request.settings_source.sha256.clone();
    let mut selected_run = original.clone();
    selected_run.system.symbol = "GBPUSD".to_owned();
    selected_run.system.base_timeframe = "H1".to_owned();
    selected_run.system.higher_timeframes = vec!["D1".to_owned()];
    assert_ne!(
        selected_run.system.symbol,
        request.settings_source.settings().system.symbol
    );
    assert_eq!(
        serde_json::to_value(request.settings_source.settings()).unwrap(),
        serde_json::to_value(&original).unwrap()
    );

    std::fs::write(
        &request.settings_source.source_path,
        b"system:\n  definitely_unknown_request_key: true\n",
    )
    .unwrap();
    assert!(DiscoverySettingsSource::load(&request.settings_source.source_path).is_err());
    assert_eq!(request.settings_source.exact_bytes, original_bytes);
    assert_eq!(request.settings_source.sha256, original_hash);
    assert!(
        request
            .settings_source
            .settings()
            .system
            .multi_resolution_prefix_base
    );
    let debug = format!("{:?}", request.settings_source);
    assert!(debug.contains(&original_hash));
    assert!(!debug.contains("multi_resolution_prefix_base"));
    assert!(!debug.contains(&original.system.symbol));
}

#[test]
fn elapsed_heartbeat_does_not_claim_work_progress_or_completed_cancellation() {
    assert_eq!(
        waiting_for_worker_message("building features", 125, false),
        "building features · awaiting engine progress — 2m 05s elapsed"
    );
    assert_eq!(
        waiting_for_worker_message("building features", 125, true),
        "building features · Stop requested; waiting for the worker to return — 2m 05s elapsed"
    );
}

#[cfg(not(feature = "gpu-nvidia"))]
fn ohlcv_buffer_addresses(data: &Ohlcv) -> [usize; 6] {
    [
        data.timestamp
            .as_ref()
            .map_or(0, |values| values.as_ptr() as usize),
        data.open.as_ptr() as usize,
        data.high.as_ptr() as usize,
        data.low.as_ptr() as usize,
        data.close.as_ptr() as usize,
        data.volume
            .as_ref()
            .map_or(0, |values| values.as_ptr() as usize),
    ]
}

#[cfg(not(feature = "gpu-nvidia"))]
#[tokio::test]
async fn owned_cpu_producer_builds_a_real_pinned_m1_feature_receipt_in_the_admitted_pool() {
    let mut request = sample_request();
    let identity = request.dataset_identity().clone();
    let start = 1_700_000_040_000_i64;
    let closes = (0..100)
        .map(|index| 1.1 + index as f64 * 0.0001 + (index as f64 * 0.3).sin() * 0.0002)
        .collect::<Vec<_>>();
    let ohlcv = Ohlcv {
        timestamp: Some((0..100).map(|index| start + index * 60_000).collect()),
        open: closes.clone(),
        high: closes.iter().map(|close| close + 0.0005).collect(),
        low: closes.iter().map(|close| close - 0.0005).collect(),
        close: closes,
        volume: None,
    };
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.app-owned-feature-fixture.v1",
        identity.canonical_bytes(),
    )
    .unwrap();
    let published = publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: &request.data_root,
        identity: &identity,
        expected_generation: Some(request.pinned_input.receipt().anchor().generation_id()),
        provenance: &provenance,
        ohlcv: &ohlcv,
        volume: CanonicalVolumeRef::Absent,
        rows_per_chunk: 100,
    })
    .unwrap();
    let selected = SelectedDatasetGenerationV1::from_manifest(published.manifest()).unwrap();
    request.pinned_input =
        Arc::new(pin_discovery_input(&request.data_root, selected.clone(), &[]).unwrap());
    let (broker, execution) = test_execution(2);
    let (tx, _rx) = mpsc::channel(8);
    let cancel = CancellationFlag::new();
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    let lease = admit_discovery_cpu_stage(
        &execution,
        &cancel,
        &tx,
        &mut snapshot,
        "test-owned feature preparation",
    )
    .await
    .unwrap();
    let feature_request = request.clone();
    let feature_broker = broker.clone();
    let input = spawn_discovery_cpu_stage(Arc::clone(&execution), lease, cancel, move |scope| {
        scope.require_current_pool()?;
        assert_eq!(scope.worker_limit().get(), 2);
        assert_eq!(BudgetedCpuExecutor::current_pool_width(), 2);
        assert_eq!(feature_broker.snapshot().live_reserved_sum, 2);
        // The real pinned decode and feature producer both run after admission.
        let dataset = feature_request
            .pinned_input
            .take_pinned_series_v1()?
            .into_cpu_dataset_without_native_adapter_v1()?;
        let original_addresses = ohlcv_buffer_addresses(&dataset.frames["M1"]);
        let original_lease = Arc::downgrade(dataset.source_artifacts["M1"].lease());
        let input =
            prepare_cpu_discovery_features(&feature_request, dataset, &[CanonicalTimeframe::M1])?;
        assert_eq!(
            ohlcv_buffer_addresses(input.base_frame().ohlcv()),
            original_addresses
        );
        assert!(Arc::ptr_eq(
            &original_lease.upgrade().expect("retained base lease"),
            input.base_frame().artifact().lease()
        ));
        Ok(input)
    })
    .await
    .unwrap()
    .expect("actual production CPU feature builder on test-owned M1 data");
    assert_eq!(broker.snapshot().available_permits, 2);
    assert_eq!(execution.executor().cached_idle_worker_threads(), 2);
    assert_eq!(input.features().n_samples(), 100);
    assert_eq!(input.base_frame().ohlcv().timestamp, ohlcv.timestamp);
    assert!(
        input
            .base_frame()
            .ohlcv()
            .close
            .iter()
            .map(|v| v.to_bits())
            .eq(ohlcv.close.iter().map(|v| v.to_bits()))
    );
    assert!(input.features().n_features() > 0);
    assert!(
        input
            .features()
            .names
            .iter()
            .all(|name| name.starts_with("M1_"))
    );
    let receipt = input.receipt().expect("seal the actual feature result");
    assert!(
        receipt
            .source_bindings()
            .iter()
            .any(|binding| binding.generation_id() == selected.generation_id())
    );
    assert_eq!(
        request
            .feature_build_options(100)
            .unwrap()
            .normalization_training_rows,
        Some(0..80)
    );
}

#[test]
fn invalid_request_fails_before_launch() {
    let mut request = sample_request();
    request.data_root = PathBuf::new();

    let err = request
        .validate()
        .expect_err("expected invalid request to fail");
    assert!(err.to_string().contains("data root"));
}

#[test]
fn duplicate_higher_timeframes_fail_instead_of_hashing_a_different_request() {
    let mut request = sample_request();
    request.higher_tfs = vec!["M5".to_owned(), "m5".to_owned()];

    let error = request
        .validate()
        .expect_err("case-normalized duplicate timeframe must fail closed");
    assert!(error.to_string().contains("duplicate higher timeframe M5"));
}

#[test]
fn a_higher_timeframe_must_be_strictly_above_the_selected_base() {
    let mut request = sample_request();
    request.higher_tfs = vec!["M1".to_owned()];

    let error = request
        .validate()
        .expect_err("base timeframe cannot also be a higher timeframe");
    assert!(error.to_string().contains("strictly above base M1"));
}

#[test]
fn request_symbol_and_base_timeframe_are_derived_from_the_pinned_receipt() {
    let request = sample_request();

    assert_eq!(request.symbol(), "EURUSD");
    assert_eq!(request.base_tf(), "M1");
    assert_eq!(
        request.dataset_identity(),
        request.pinned_input.receipt().anchor().identity(),
        "request selectors must come only from the pinned receipt"
    );
}

#[test]
fn run_settings_reject_config_from_a_different_selected_series() {
    for changed in ["symbol", "timeframe", "higher_timeframes"] {
        let mut request = sample_request();
        let config = request.config.as_mut().expect("resolved test config");
        match changed {
            "symbol" => config.evaluation_symbol = "GBPUSD".to_owned(),
            "timeframe" => config.timeframe_label = "H1".to_owned(),
            "higher_timeframes" => config.higher_timeframes = vec!["H4".to_owned()],
            _ => unreachable!(),
        }
        assert!(
            request.validate().is_err(),
            "an already-costed config with different {changed} must be refused before launch",
        );
    }
}

#[test]
fn run_settings_keep_the_once_resolved_risk_and_frequency_policy() {
    let (root, request) = request_for(CanonicalTimeframe::H1, Vec::new());
    let mut request = TestRequest {
        root,
        request: Some(request),
    };
    let config = request.config.as_mut().expect("resolved test config");
    config.mode = neoethos_search::discovery::DiscoveryMode::PropFirm;
    config.filtering.min_trades_per_month = 20.0;
    config.filtering.opportunistic_min_trades_per_month = 10.0;
    config.prop_firm_risk_band = Some((0.002, 0.008));
    config.evaluation_spread_pips = 2.5;
    config.evaluation_commission_per_trade = 14.0;
    config.population = 200;
    config.generations = 1_000;
    // The Settings adapter has ALREADY resolved the mode. This fixture tests
    // only that hand-off, without authorizing a broker evaluation or search.
    request.config = Some(
        request
            .config
            .take()
            .expect("resolved test config")
            .apply_mode_overrides(),
    );
    let expected = request
        .config
        .as_ref()
        .expect("resolved test config")
        .clone();
    assert_eq!(expected.filtering.min_trades_per_month, 8.0); // 20 * H1's 0.4
    assert_eq!(expected.filtering.opportunistic_min_trades_per_month, 4.0);

    request.validate().expect("exact run selection");
    let worker_config = request.execution_config().expect("resolved worker config");
    assert_eq!(worker_config.filtering.min_trades_per_month, 8.0);
    assert_eq!(
        worker_config.filtering.opportunistic_min_trades_per_month,
        4.0
    );
    assert_eq!(worker_config.risk_per_trade_min, 0.002);
    assert_eq!(worker_config.risk_per_trade_max, 0.008);
    assert_eq!(format!("{worker_config:?}"), format!("{expected:?}"));
}

#[test]
fn discovery_requires_only_base_and_explicit_higher_timeframes() {
    let (h4_root, h4_request) = request_for(CanonicalTimeframe::H4, Vec::new());
    assert_eq!(
        required_direct_timeframes(&h4_request).expect("base-only direct set"),
        vec![CanonicalTimeframe::H4]
    );
    drop(h4_request);
    std::fs::remove_dir_all(h4_root).expect("remove H4 test root");

    let (m15_root, m15_request) = request_for(CanonicalTimeframe::M15, vec!["H1".to_owned()]);
    assert_eq!(
        required_direct_timeframes(&m15_request).expect("base plus explicit higher direct set"),
        vec![CanonicalTimeframe::M15, CanonicalTimeframe::H1]
    );
    drop(m15_request);
    std::fs::remove_dir_all(m15_root).expect("remove M15/H1 test root");
}

#[test]
fn pinned_input_ignores_another_legitimate_source_and_survives_pointer_advance() {
    let root = unique_test_root("exact-discovery");
    std::fs::create_dir_all(&root).expect("create exact-selection test root");

    let selected = CanonicalDatasetIdentity::external(
        "selected-source",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("selected identity");
    let other = CanonicalDatasetIdentity::external(
        "other-source",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("other identity");
    let generation_one = publish_two_bar_fixture(&root, &selected, 1.125, None);
    publish_two_bar_fixture(&root, &other, 9.875, None);

    let pinned =
        pin_discovery_input(&root, generation_one.clone(), &[]).expect("pin selected generation");

    publish_two_bar_fixture(
        &root,
        &selected,
        1.250,
        Some(generation_one.generation_id()),
    );
    #[cfg(not(feature = "gpu-nvidia"))]
    {
        let dataset = pinned
            .take_pinned_series_v1()
            .expect("move exact pinned series")
            .into_cpu_dataset_without_native_adapter_v1()
            .expect("decode the reader-leased generation after pointer advance");
        assert_eq!(dataset.frames["M1"].close, vec![1.125, 1.125]);
        assert_eq!(dataset.source_artifacts["M1"].identity(), &selected);
    }
    #[cfg(feature = "gpu-nvidia")]
    assert_eq!(
        pinned.receipt().anchor().generation_id(),
        generation_one.generation_id(),
        "the CUDA build retains the exact leased generation until a sealed route selects its factory"
    );

    let stale = pin_discovery_input(&root, generation_one, &[])
        .expect_err("a new run with the stale receipt must conflict");
    assert!(
        stale
            .downcast_ref::<neoethos_data::ExactDatasetGenerationConflict>()
            .is_some(),
        "stale receipt must remain a typed conflict: {stale:#}"
    );

    drop(pinned);
    std::fs::remove_dir_all(&root).expect("remove exact-selection test root");
}

#[test]
fn background_identity_resolution_rejects_ambiguity_and_lists_every_candidate() {
    let first = CanonicalDatasetIdentity::external(
        "source-a",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("first identity");
    let second = CanonicalDatasetIdentity::external(
        "source-b",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("second identity");

    let error =
        select_unique_background_identity(vec![second.clone(), first.clone()], "EURUSD", "M1")
            .expect_err("background selection must not pick first when two series match");
    let message = error.to_string();
    assert!(message.contains(&first.to_path_component()));
    assert!(message.contains(&second.to_path_component()));
}

#[test]
fn background_identity_resolution_rejects_zero_exact_matches_and_lists_known_series() {
    let m5 = CanonicalDatasetIdentity::external(
        "source-a",
        "EURUSD",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("M5 identity");
    let h1 = CanonicalDatasetIdentity::external(
        "source-b",
        "EURUSD",
        CanonicalTimeframe::H1,
        BarTimestampConvention::BarOpen,
    )
    .expect("H1 identity");

    let error = select_unique_background_identity(vec![h1.clone(), m5.clone()], "EURUSD", "M1")
        .expect_err("background selection must not choose another timeframe");
    let message = error.to_string();
    assert!(message.contains(&m5.to_path_component()));
    assert!(message.contains(&h1.to_path_component()));
}

#[test]
fn background_identity_resolution_returns_the_only_exact_match() {
    let selected = CanonicalDatasetIdentity::external(
        "source-a",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("selected identity");
    let other_timeframe =
        identity_for_timeframe(&selected, CanonicalTimeframe::H1).expect("same series H1 identity");

    let resolved =
        select_unique_background_identity(vec![other_timeframe, selected.clone()], "eurusd", "m1")
            .expect("one exact background identity");
    assert_eq!(resolved, selected);
}

#[test]
fn target_timeframe_identity_preserves_the_exact_source_or_broker_scope() {
    let request = sample_request();
    let external = request.dataset_identity().clone();
    let external_h1 = identity_for_timeframe(&external, CanonicalTimeframe::H1)
        .expect("derive exact external H1 identity");
    assert_eq!(external_h1.scope(), external.scope());
    assert_eq!(external_h1.symbol_name(), external.symbol_name());
    assert_eq!(external_h1.timeframe(), CanonicalTimeframe::H1);

    let broker = CanonicalDatasetIdentity::ctrader(
        neoethos_data::CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        42,
        1,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("broker identity");
    let broker_h4 = identity_for_timeframe(&broker, CanonicalTimeframe::H4)
        .expect("derive exact broker H4 identity");
    assert_eq!(broker_h4.scope(), broker.scope());
    assert_eq!(broker_h4.symbol_name(), broker.symbol_name());
    assert_eq!(broker_h4.timeframe(), CanonicalTimeframe::H4);
}

#[test]
fn required_direct_timeframe_set_contains_only_base_and_explicit_higher_frames() {
    let mut request = sample_request();
    request.higher_tfs = vec!["M5".to_owned(), "H1".to_owned(), "H4".to_owned()];

    let required = required_direct_timeframes(&request).expect("canonical direct timeframe set");
    assert_eq!(
        required,
        vec![
            CanonicalTimeframe::M1,
            CanonicalTimeframe::M5,
            CanonicalTimeframe::H1,
            CanonicalTimeframe::H4,
        ]
    );
}

#[test]
fn cancellation_request_maps_to_cancelled_snapshot() {
    let snapshot = cancelled_snapshot(JobKind::Discovery, "operator cancelled discovery");

    assert_eq!(snapshot.state, JobState::Cancelled);
    assert_eq!(snapshot.report.summary, "operator cancelled discovery");
}

#[test]
fn empty_portfolio_failure_maps_to_failed_snapshot() {
    let snapshot = failed_snapshot(
        JobKind::Discovery,
        anyhow::anyhow!("Discovery produced an empty portfolio for EURUSD M1 (candidates=4)"),
    );

    assert_eq!(snapshot.state, JobState::Failed);
    assert_eq!(snapshot.report.errors.len(), 1);
    assert!(snapshot.report.errors[0].contains("empty portfolio"));
}

#[test]
fn success_snapshot_carries_candidate_and_portfolio_counters() {
    let best = Gene {
        strategy_id: "alpha-1".to_string(),
        fitness: 1450.0,
        sharpe_ratio: 1.82,
        win_rate: 0.64,
        ..Gene::default()
    };

    let second = Gene {
        strategy_id: "alpha-2".to_string(),
        fitness: 1200.0,
        sharpe_ratio: 1.55,
        win_rate: 0.59,
        ..Gene::default()
    };

    let (search_input_receipt, selection_scope) = sample_discovery_authority();
    let result = DiscoveryResult {
        search_input_receipt,
        selection_scope,
        calibration_scope: None,
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_census: neoethos_search::discovery::CostBandCensus {
            survives: 0,
            optimistic_edge_only: 0,
            fails: 0,
            unmeasured: 0,
            not_discriminating: 0,
        },
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![best.clone(), second],
        candidates: vec![best, Gene::default(), Gene::default()],
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let snapshot = completed_snapshot(JobSnapshot::new(JobKind::Discovery), &result);

    assert_eq!(snapshot.state, JobState::Succeeded);
    assert_eq!(
        snapshot.report.counters,
        vec![
            ("candidates".to_string(), 3),
            ("portfolio".to_string(), 2),
            ("not_selected".to_string(), 1),
            ("quality_scored".to_string(), 0),
            ("trade_logs".to_string(), 0),
        ]
    );
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_strategy" && value == "alpha-1" })
    );
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_sharpe" && value == "1.82" })
    );
    assert!(
        snapshot
            .report
            .entries
            .iter()
            .any(|entry| entry.contains("alpha-1") && entry.contains("win_rate=0.64"))
    );
    assert!(
        snapshot
            .report
            .events
            .iter()
            .any(|event| event.message.contains("completed discovery"))
    );
}

// #211: the `completed_snapshot` highlight emits `best_oos_sharpe`
// taken from `forward_test_validation_artifacts`, distinct from
// `best_sharpe` (which is in-sample stage-1). Both columns end up in
// the validation CSV so a big IS-OOS gap can be spotted at-a-glance.
#[test]
fn success_snapshot_emits_best_oos_sharpe_from_forward_test_artifacts() {
    use neoethos_search::{
        BacktestMetrics, CanonicalSearchArtifactScopeV2, CanonicalSearchWindowRoleV1,
        ForwardTestSummary, ForwardTestValidationArtifactFile,
    };

    let best = Gene {
        strategy_id: "alpha-1".to_string(),
        fitness: 1450.0,
        // In-sample stage-1 Sharpe — the GA optimized for this.
        sharpe_ratio: 5.50,
        win_rate: 0.64,
        ..Gene::default()
    };

    let lo_oos_metrics = BacktestMetrics {
        net_profit: 0.0,
        sharpe: 1.20,
        peak_equity: 0.0,
        max_drawdown: 0.0,
        win_rate: 0.0,
        profit_factor: 0.0,
        expectancy: 0.0,
        monthly_target_hit_rate: 0.0,
        trade_count: 0,
        consistency: 0.0,
        max_daily_drawdown: 0.0,
    };
    let hi_oos_metrics = BacktestMetrics {
        sharpe: 1.55,
        ..lo_oos_metrics
    };

    let (search_input_receipt, selection_scope) = sample_discovery_authority();
    let holdout_scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
        CanonicalSearchWindowRoleV1::Holdout,
        search_input_receipt.clone(),
    )
    .expect("canonical full holdout test scope");
    let holdout_bars = (holdout_scope.evaluated_window().row_end()
        - holdout_scope.evaluated_window().row_start()) as usize;
    let search_config_hash = "fnv64:0123456789abcdef";

    let forward_artifacts = vec![
        ForwardTestValidationArtifactFile::new(
            holdout_scope.clone(),
            search_config_hash,
            &best,
            ForwardTestSummary {
                bars: holdout_bars,
                metrics: lo_oos_metrics,
                span_days: 1.0,
            },
        )
        .expect("lower-Sharpe canonical forward-test artifact"),
        ForwardTestValidationArtifactFile::new(
            holdout_scope.clone(),
            search_config_hash,
            &best,
            ForwardTestSummary {
                bars: holdout_bars,
                metrics: hi_oos_metrics,
                span_days: 1.0,
            },
        )
        .expect("higher-Sharpe canonical forward-test artifact"),
    ];

    let result = DiscoveryResult {
        search_input_receipt,
        selection_scope,
        calibration_scope: None,
        holdout_scope: Some(holdout_scope),
        search_config_hash: search_config_hash.to_string(),
        cost_band_census: neoethos_search::discovery::CostBandCensus {
            survives: 0,
            optimistic_edge_only: 0,
            fails: 0,
            unmeasured: 0,
            not_discriminating: 0,
        },
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![best.clone()],
        candidates: vec![best],
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: forward_artifacts,
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let snapshot = completed_snapshot(JobSnapshot::new(JobKind::Discovery), &result);

    // In-sample Sharpe still emitted (unchanged from prior contract).
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_sharpe" && value == "5.50" }),
        "best_sharpe (in-sample) must still be present"
    );
    // New OOS highlight picks the MAX Sharpe across the forward-test
    // tail artifacts — 1.55 wins over 1.20.
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_oos_sharpe" && value == "1.5500" }),
        "best_oos_sharpe must be the max forward-test sharpe (1.55)"
    );
}

#[test]
fn success_snapshot_omits_best_oos_sharpe_when_forward_test_artifacts_empty() {
    // Backward compatibility: when no forward-test artifacts are
    // produced (tail too short, or `compute_discovery_forward_test_artifacts`
    // failed) the highlight is simply absent. The validation harness
    // treats absence as `None` and falls back to in-sample reporting.
    let best = Gene {
        strategy_id: "alpha-1".to_string(),
        sharpe_ratio: 1.82,
        ..Gene::default()
    };
    let (search_input_receipt, selection_scope) = sample_discovery_authority();
    let result = DiscoveryResult {
        search_input_receipt,
        selection_scope,
        calibration_scope: None,
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_census: neoethos_search::discovery::CostBandCensus {
            survives: 0,
            optimistic_edge_only: 0,
            fails: 0,
            unmeasured: 0,
            not_discriminating: 0,
        },
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![best.clone()],
        candidates: vec![best],
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };
    let snapshot = completed_snapshot(JobSnapshot::new(JobKind::Discovery), &result);
    assert!(
        !snapshot
            .report
            .highlights
            .iter()
            .any(|(name, _)| name == "best_oos_sharpe"),
        "best_oos_sharpe must be absent when no forward-test artifacts exist"
    );
    // best_sharpe (in-sample) is still emitted.
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, _)| name == "best_sharpe")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn terminal_delivery_waits_for_capacity_and_preserves_all_final_states() {
    for state in [
        JobState::Succeeded,
        JobState::Degraded,
        JobState::Failed,
        JobState::Cancelled,
    ] {
        let (tx, mut rx) = mpsc::channel(1);
        let mut progress = JobSnapshot::new(JobKind::Discovery);
        progress.state = JobState::Running;
        progress.progress.message = "last intermediate snapshot".to_owned();
        tx.try_send(ServiceEvent::DiscoveryUpdated(progress.clone()))
            .unwrap();

        let mut terminal = progress.clone();
        terminal.state = state;
        terminal.report.summary = format!("actual terminal outcome: {state:?}");
        terminal.report.counters.push(("evaluated".to_owned(), 37));
        terminal
            .report
            .errors
            .push("retained diagnostic".to_owned());
        terminal.report.highlights.push((
            "promotion_eligibility".to_owned(),
            "NotPromotionEligible".to_owned(),
        ));
        let publication = send_terminal_snapshot(&tx, &terminal);
        tokio::pin!(publication);
        assert!(
            futures::poll!(&mut publication).is_pending(),
            "a full queue must suspend delivery, not discard the terminal outcome"
        );

        let Some(ServiceEvent::DiscoveryUpdated(first)) = rx.recv().await else {
            panic!("queued progress is retained");
        };
        assert_eq!(first, progress);
        tokio::time::timeout(std::time::Duration::from_secs(2), &mut publication)
            .await
            .expect("delivery resumes when the consumer frees capacity");
        let Some(ServiceEvent::DiscoveryUpdated(delivered)) = rx.recv().await else {
            panic!("terminal snapshot was lost");
        };
        assert_eq!(delivered, terminal);
        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn terminal_delivery_finishes_when_receiver_was_already_closed() {
    let (tx, rx) = mpsc::channel(1);
    drop(rx);
    let terminal = failed_snapshot(JobKind::Discovery, anyhow::anyhow!("original failure"));
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        send_terminal_snapshot(&tx, &terminal),
    )
    .await
    .expect("disconnection must not strand terminal delivery");
    assert_eq!(terminal.report.summary, "original failure");
}

#[tokio::test(flavor = "current_thread")]
async fn terminal_delivery_finishes_when_receiver_closes_during_backpressure() {
    let (tx, rx) = mpsc::channel(1);
    tx.try_send(ServiceEvent::DiscoveryUpdated(JobSnapshot::new(
        JobKind::Discovery,
    )))
    .unwrap();
    let terminal = cancelled_snapshot(JobKind::Discovery, "operator stopped the run");
    let publication = send_terminal_snapshot(&tx, &terminal);
    tokio::pin!(publication);
    assert!(futures::poll!(&mut publication).is_pending());
    drop(rx);
    tokio::time::timeout(std::time::Duration::from_secs(2), &mut publication)
        .await
        .expect("a disconnected receiver releases an already-waiting publisher");
}

#[test]
fn progress_delivery_remains_nonblocking_when_the_queue_is_full() {
    let (tx, mut rx) = mpsc::channel(1);
    let mut first = JobSnapshot::new(JobKind::Discovery);
    first.state = JobState::Running;
    tx.try_send(ServiceEvent::DiscoveryUpdated(first.clone()))
        .unwrap();
    let mut later = first.clone();
    later.progress.message = "replaceable progress".to_owned();
    try_send_progress_event(&tx, ServiceEvent::DiscoveryUpdated(later));
    let ServiceEvent::DiscoveryUpdated(delivered) = rx.try_recv().unwrap() else {
        panic!("expected Discovery progress");
    };
    assert_eq!(delivered, first);
    assert!(matches!(
        rx.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn terminal_delivery_from_real_job_retains_stop_when_initial_snapshot_fills_queue() {
    let mut request = sample_request();
    request.config = None;
    let (broker, execution) = test_execution(2);
    let (tx, mut rx) = mpsc::channel(1);
    let handle = start_discovery_job(request.clone(), Arc::clone(&execution), tx)
        .expect("start the actual Discovery owner");
    // current_thread has not polled the spawned worker. Stop precedes feature
    // preparation/broker work while the synchronous initial event fills the queue.
    handle.cancel.request();
    tokio::task::yield_now().await;
    let snapshots = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        let mut snapshots = Vec::new();
        while let Some(ServiceEvent::DiscoveryUpdated(snapshot)) = rx.recv().await {
            snapshots.push(snapshot);
        }
        snapshots
    })
    .await
    .expect("actual worker must finish after the UI consumer resumes");

    assert_eq!(
        snapshots.len(),
        2,
        "initial plus exactly one terminal event"
    );
    assert_eq!(snapshots[0].state, JobState::Running);
    assert_eq!(snapshots[1].state, JobState::Cancelled);
    assert_eq!(snapshots[0].id, snapshots[1].id);
    assert!(
        snapshots[1]
            .report
            .summary
            .contains("before feature preparation")
    );
    assert!(request.pinned_input.pinned_series.lock().unwrap().is_some());
    assert_eq!(broker.snapshot().live_reserved_sum, 0);
    assert_eq!(execution.executor().cached_idle_worker_threads(), 0);
}

#[tokio::test]
async fn start_discovery_job_emits_initial_snapshot_with_requested_targets() {
    let higher_tfs = vec!["M5".to_string(), "M15".to_string(), "H1".to_string()];
    let (root, mut request) = request_for(CanonicalTimeframe::M1, higher_tfs);
    // Admission now carries requested dimensions without a financially usable
    // config. This test cancels before the worker can contact a broker.
    request.config = None;
    request.overrides = TypedDiscoveryOverridesV1::checked_new(
        Some(96),
        Some(TypedDiscoveryGenerationOverrideV1::Exact(7)),
        None,
        None,
        Some(144),
        Some(24),
    )
    .expect("requested test overrides");
    let (tx, mut rx) = mpsc::channel(10000);

    let (_broker, execution) = test_execution(2);
    let handle = start_discovery_job(request.clone(), execution, tx).expect("job should start");
    let event = rx.recv().await.expect("expected initial discovery event");
    let ServiceEvent::DiscoveryUpdated(snapshot) = event else {
        panic!("expected discovery update event");
    };

    assert_eq!(snapshot.state, JobState::Running);
    assert_eq!(snapshot.progress.stage, "using_pinned_data");
    assert_eq!(
        snapshot.report.counters,
        vec![
            ("target_candidates".to_string(), 144),
            ("target_portfolio".to_string(), 24),
            ("generations".to_string(), 7),
            ("population".to_string(), 96),
            ("planned_ga_evaluations".to_string(), 672),
        ]
    );
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| name == "symbol" && value == "EURUSD")
    );
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| name == "higher_tfs" && value == "M5, M15, H1")
    );
    assert!(snapshot.report.events.iter().any(|event| {
        event.message.contains("planned discovery")
            && event.message.contains("candidate_count=144")
            && event.message.contains("portfolio_size=24")
    }));
    assert_eq!(
        snapshot.report.log_path,
        Some(canonical_log_path().display().to_string())
    );

    handle.cancel.request();
    drop(request);
    loop {
        let next = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("discovery cancellation timed out");
        let Some(ServiceEvent::DiscoveryUpdated(update)) = next else {
            break;
        };
        if matches!(
            update.state,
            JobState::Cancelled | JobState::Failed | JobState::Succeeded | JobState::Degraded
        ) {
            break;
        }
    }
    drop(handle);
    std::fs::remove_dir_all(root).expect("remove start-job test root");
}

#[tokio::test(flavor = "current_thread")]
async fn discovery_cpu_admission_waits_for_the_installed_width_without_blocking_tokio() {
    let (broker, execution) = test_execution(3);
    let held = execution
        .admission_client()
        .admit(CpuPermitRequest::local(WorkerLimit::new(1).unwrap()))
        .await
        .unwrap();
    let (tx, mut rx) = mpsc::channel(8);
    let cancel = CancellationFlag::new();
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    let lease = {
        let admission = admit_discovery_cpu_stage(
            &execution,
            &cancel,
            &tx,
            &mut snapshot,
            "test strategy search",
        );
        tokio::pin!(admission);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), &mut admission,)
                .await
                .is_err(),
            "two free permits must not silently narrow a three-worker run"
        );
        assert_eq!(broker.snapshot().available_permits, 2);
        assert_eq!(execution.executor().cached_idle_worker_threads(), 0);
        let Some(ServiceEvent::DiscoveryUpdated(waiting)) = rx.recv().await else {
            panic!("expected a CPU wait event");
        };
        assert_eq!(waiting.progress.stage, "waiting_for_cpu");
        assert!(waiting.progress.percent.is_none());
        assert!(
            waiting
                .report
                .counters
                .contains(&("cpu_workers_requested".to_owned(), 3))
        );
        assert!(
            waiting
                .report
                .counters
                .contains(&("cpu_workers_reserved".to_owned(), 0))
        );
        drop(held);
        tokio::time::timeout(std::time::Duration::from_secs(2), &mut admission)
            .await
            .unwrap()
            .unwrap()
    };
    assert_eq!(lease.width().get(), 3);
    assert!(
        snapshot
            .report
            .counters
            .contains(&("cpu_workers_reserved".to_owned(), 3))
    );
    let stage_broker = broker.clone();
    let observed_width = spawn_discovery_cpu_stage(execution, lease, cancel, move |scope| {
        scope.require_current_pool()?;
        assert_eq!(stage_broker.snapshot().live_reserved_sum, 3);
        assert!(matches!(
            stage_broker.try_acquire(CpuPermitRequest::local(WorkerLimit::new(1).unwrap())),
            Err(AcquireError::NestedAcquisition)
        ));
        Ok((
            scope.worker_limit().get(),
            BudgetedCpuExecutor::current_pool_width(),
        ))
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(observed_width, (3, 3));
    assert_eq!(broker.snapshot().available_permits, 3);
}

#[tokio::test(flavor = "current_thread")]
async fn discovery_cpu_stage_skips_work_cancelled_after_admission() {
    let (broker, execution) = test_execution(2);
    let (tx, _rx) = mpsc::channel(8);
    let cancel = CancellationFlag::new();
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    let lease = admit_discovery_cpu_stage(
        &execution,
        &cancel,
        &tx,
        &mut snapshot,
        "cancel-before-start test",
    )
    .await
    .unwrap();
    cancel.request();
    let result = spawn_discovery_cpu_stage(execution, lease, cancel, |_| -> Result<()> {
        panic!("cancelled blocking work must not start");
    })
    .await
    .expect("cancelled work is a normal result, not a panic");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("__DISCOVERY_CANCELLED__")
    );
    assert_eq!(broker.snapshot().available_permits, 2);
}

#[tokio::test(flavor = "current_thread")]
async fn discovery_cpu_stage_returns_capacity_after_error_and_panic() {
    let (broker, execution) = test_execution(2);
    let (tx, _rx) = mpsc::channel(8);
    let cancel = CancellationFlag::new();
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    for panic_in_worker in [false, true] {
        let lease = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            admit_discovery_cpu_stage(
                &execution,
                &cancel,
                &tx,
                &mut snapshot,
                "failure-cleanup test",
            ),
        )
        .await
        .unwrap()
        .unwrap();
        let result = spawn_discovery_cpu_stage(
            Arc::clone(&execution),
            lease,
            cancel.clone(),
            move |scope| -> Result<()> {
                scope.require_current_pool()?;
                if panic_in_worker {
                    panic!("synthetic Discovery CPU worker panic");
                }
                anyhow::bail!("synthetic Discovery CPU worker error");
            },
        )
        .await;
        if panic_in_worker {
            assert!(result.unwrap_err().is_panic());
        } else {
            assert!(
                result
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("synthetic Discovery")
            );
        }
        assert_eq!(broker.snapshot().available_permits, 2);
        assert_eq!(broker.snapshot().live_reserved_sum, 0);
    }
    let next = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        execution
            .admission_client()
            .admit(CpuPermitRequest::local(WorkerLimit::new(2).unwrap())),
    )
    .await
    .unwrap()
    .unwrap();
    drop(next);
    assert_eq!(broker.snapshot().available_permits, 2);
}

#[tokio::test(flavor = "current_thread")]
async fn discovery_job_stop_cancels_cpu_wait_without_consuming_its_pinned_data() {
    let mut request = sample_request();
    request.config = None;
    let (broker, execution) = test_execution(2);
    let held = execution
        .admission_client()
        .admit(CpuPermitRequest::local(WorkerLimit::new(1).unwrap()))
        .await
        .unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    let handle = start_discovery_job(request.clone(), Arc::clone(&execution), tx).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let Some(ServiceEvent::DiscoveryUpdated(snapshot)) = rx.recv().await else {
                panic!("Discovery exited before CPU admission");
            };
            assert_eq!(snapshot.state, JobState::Running);
            if snapshot.progress.stage == "waiting_for_cpu" {
                assert!(snapshot.progress.percent.is_none());
                assert!(
                    snapshot
                        .report
                        .counters
                        .contains(&("cpu_workers_requested".to_owned(), 2))
                );
                break;
            }
        }
    })
    .await
    .unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
            .await
            .is_err(),
        "Discovery must not start its producer while the full CPU request is unavailable"
    );
    assert!(request.pinned_input.pinned_series.lock().unwrap().is_some());
    assert_eq!(execution.executor().cached_idle_worker_threads(), 0);
    handle.cancel.request();
    let mut terminal = None;
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while let Some(ServiceEvent::DiscoveryUpdated(snapshot)) = rx.recv().await {
            if !matches!(snapshot.state, JobState::Queued | JobState::Running) {
                terminal = Some(snapshot);
            }
        }
    })
    .await
    .expect("Stop must finish without waiting for the unrelated held permit");
    let terminal = terminal.expect("real Discovery worker emits terminal evidence");
    assert_eq!(terminal.state, JobState::Cancelled);
    assert!(
        terminal
            .report
            .summary
            .contains("waiting for input-preparation CPU")
    );
    assert!(request.pinned_input.pinned_series.lock().unwrap().is_some());
    assert_eq!(broker.snapshot().live_reserved_sum, 1);
    assert_eq!(execution.executor().cached_idle_worker_threads(), 0);
    drop(held);
    let next = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        execution
            .admission_client()
            .admit(CpuPermitRequest::local(WorkerLimit::new(2).unwrap())),
    )
    .await
    .unwrap()
    .unwrap();
    drop(next);
    assert_eq!(broker.snapshot().available_permits, 2);
}

#[test]
fn backend_portfolio_milestone_updates_discovery_snapshot_with_live_counts() {
    let request = sample_request();
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    snapshot.state = JobState::Running;
    snapshot.progress = JobProgress {
        percent: Some(0.75),
        stage: "running_discovery".to_string(),
        message: "evaluating strategy candidates for EURUSD".to_string(),
    };
    snapshot.report = JobReport {
        counters: requested_discovery_counters(&request),
        highlights: requested_discovery_highlights(&request),
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };

    apply_backend_discovery_event(
        &mut snapshot,
        &neoethos_search::DiscoveryProgress::PortfolioSelected {
            portfolio_size: 12,
            rejected_by_correlation: 5,
            target_portfolio: 24,
        },
    );

    assert_eq!(snapshot.state, JobState::Running);
    assert_eq!(snapshot.progress.stage, "portfolio_construction");
    assert!(snapshot.progress.percent.expect("percent should exist") >= 0.9);
    assert!(
        snapshot
            .report
            .counters
            .iter()
            .any(|(name, value)| name == "portfolio" && *value == 12)
    );
    assert!(
        snapshot
            .report
            .counters
            .iter()
            .any(|(name, value)| name == "rejected_by_correlation" && *value == 5)
    );
    assert!(
        snapshot
            .report
            .events
            .iter()
            .any(|event| event.message.contains("portfolio selection"))
    );
    assert!(
        snapshot
            .report
            .entries
            .iter()
            .any(|entry| entry.contains("portfolio | accepted=12"))
    );
}

#[cfg(not(feature = "gpu-nvidia"))]
#[tokio::test(flavor = "current_thread")]
async fn two_disjoint_cpu_batches_save_exact_recipes_replay_and_honor_stop() {
    let request = broker_research_request(true, "USD");
    let (_broker, execution) = test_execution(2);
    let cancel = CancellationFlag::new();
    let (tx, _rx) = mpsc::channel(8);
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    let lease = admit_discovery_cpu_stage(
        &execution,
        &cancel,
        &tx,
        &mut snapshot,
        "two real feature batches and saved replay",
    )
    .await
    .unwrap();
    spawn_discovery_cpu_stage(execution, lease, cancel.clone(), move |scope| {
        scope.require_current_pool()?;
        let dataset = request.pinned_input.take_pinned_series_v1()?
            .into_cpu_dataset_without_native_adapter_v1()?;
        let observations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = Arc::clone(&observations);
        let control = FeatureBuildControl::new(cancel.cancel_arc())
            .with_indicator_compute_policy(neoethos_data::IndicatorComputePolicy::CpuOnly)
            .with_observer(move |_| { observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst); });
        let mut cursor = 0;
        let mut prior_batch = None;
        let mut identities = Vec::new();
        // Test-only width bounds fixture cost; the production planner uses measured RAM.
        for number in 0..2 {
            let batch = neoethos_data::search_working_set_batch_seeded(cursor, 2, true, 0x9385);
            assert!(batch.next_cursor > cursor);
            if let Some(previous) = &prior_batch { assert_ne!(previous, &batch); }
            let next_cursor = batch.next_cursor;
            let input = prepare_cpu_discovery_batch_with_control(&request, &dataset,
                &required_direct_timeframes(&request)?, Arc::new(batch.clone()), &control)?;
            let receipt = input.receipt().map_err(anyhow::Error::new)?;
            assert_eq!(receipt.feature_execution().compute_policy(),
                neoethos_search::data_selection::CanonicalFeatureComputePolicyV1::CpuOnly);
            let options = receipt.feature_build_options().expect("actual persisted recipe").clone();
            assert_eq!(options.classic_ta_working_set.as_ref(), Some(&batch));
            assert!(requested_discovery_counters(&request).contains(&("population".into(), 96)));
            let path = request.data_root.join(format!("batch-{number}.receipt.json"));
            std::fs::write(&path, receipt.to_json_bytes().map_err(anyhow::Error::new)?)?;
            let saved = CanonicalSearchInputReceiptV2::from_json_bytes(&std::fs::read(&path)?)
                .map_err(anyhow::Error::new)?;
            assert_eq!(saved.feature_execution().compute_policy(),
                neoethos_search::data_selection::CanonicalFeatureComputePolicyV1::CpuOnly);
            let identity = saved.identity_sha256().map_err(anyhow::Error::new)?;
            drop(input);
            let replay = neoethos_search::data_selection::CanonicalSearchInput::from_recorded_receipt_with_control(
                &request.data_root, saved, &options, &control,
            ).map_err(anyhow::Error::new)?;
            assert_eq!(replay.receipt().map_err(anyhow::Error::new)?.identity_sha256().map_err(anyhow::Error::new)?, identity);
            replay.as_run_input().map_err(anyhow::Error::new)?;
            identities.push(identity);
            prior_batch = Some(batch);
            cursor = next_cursor;
        }
        assert_ne!(identities[0], identities[1]);
        assert!(observations.load(std::sync::atomic::Ordering::SeqCst) > 0);
        cancel.request();
        let before = observations.load(std::sync::atomic::Ordering::SeqCst);
        let third = neoethos_data::search_working_set_batch_seeded(cursor, 2, true, 0x9385);
        assert!(prepare_cpu_discovery_batch_with_control(&request, &dataset,
            &required_direct_timeframes(&request)?, Arc::new(third), &control).is_err());
        assert_eq!(observations.load(std::sync::atomic::Ordering::SeqCst), before);
        assert!(request.data_root.join("batch-0.receipt.json").is_file());
        assert!(request.data_root.join("batch-1.receipt.json").is_file());
        Ok(())
    }).await.unwrap().unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn discovery_overall_deadline_is_not_reset_between_batches_and_retains_saved_counts() {
    let cancel = CancellationFlag::new();
    let mut deadline = DiscoveryDeadline::default();
    deadline.start_once(0.001 / 3600.0, cancel.clone()).unwrap();
    deadline.start_once(24.0, cancel.clone()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !cancel.is_requested() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the second batch cannot extend the first deadline");
    let progress = DiscoveryWorkingSetProgress {
        active: 2,
        completed: 1,
        completed_entries: 3,
        total_entries: 20,
        ..Default::default()
    };
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    progress.apply(&mut snapshot.report, u64::MAX);
    apply_backend_discovery_event(
        &mut snapshot,
        &DiscoveryProgress::SearchStarted {
            population: 200,
            generations: 1000,
            max_indicators: 8,
        },
    );
    let stopped = deadline.cancelled(snapshot, "operator Stop");
    assert_eq!(stopped.state, JobState::Cancelled);
    assert_eq!(stopped.progress.stage, "overall_time_budget_exhausted");
    assert!(
        stopped
            .report
            .counters
            .contains(&("working_set_completed_entries".into(), 3))
    );
    assert!(
        stopped
            .report
            .counters
            .contains(&("working_set_saved_results".into(), 1))
    );
    assert!(
        stopped
            .report
            .highlights
            .contains(&("working_set_seed".into(), u64::MAX.to_string()))
    );
}
