#![cfg(all(feature = "gpu-cuda", target_os = "linux"))]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use neoethos_core::Settings;
use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::core::features::FeatureBuildOptions;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1,
    CanonicalOhlcvPublishRequest, CanonicalTimeframe, CanonicalVolumeRef, Ohlcv,
    SelectedDatasetGenerationV1, publish_canonical_ohlcv_generation,
};
use neoethos_search::data_selection::CanonicalSearchInput;
use neoethos_search::{
    CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1,
    CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1,
    CanonicalNativeCancellationTokenV1, CanonicalNativeDiscoveryExecutionErrorCodeV1,
    CanonicalNativeDiscoveryExecutionStageV1, CanonicalNativeGenerationZeroOverridesV1,
    CanonicalNativeRuntimeInstallReceiptV1, CanonicalResearchContractArtifactRefV1,
    CanonicalTrendbarResearchCostAssumptionsV2, CanonicalTrendbarResearchExecutionContractV3,
    DiscoveryProgress, PublishedCanonicalNativeGenerationZeroResearchV1,
    install_and_seal_canonical_native_runtime_authority_v1,
    run_canonical_native_discovery_generation_zero_from_ref_v1, set_migration_enabled,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const RUN_ENV: &str = "NEOETHOS_RUN_CANONICAL_NATIVE_DISCOVERY_V1_DEVICE_TEST";
const RESULT_IDENTITY_DOMAIN_V1: &[u8] =
    b"neoethos.canonical-native-generation-zero-research-result.identity.v1\0";
const HAPPY_CONTRACT: &str = "research/contracts/canonical-native-device-happy.json";
const STALE_CONTRACT: &str = "research/contracts/canonical-native-device-stale.json";
const M15_MILLIS: i64 = 15 * 60 * 1_000;
const FIXTURE_ROWS: usize = 1_200;
const CONFIGURED_POPULATION: usize = 200;
const MAX_INDICATORS: usize = 5;
const NATIVE_AUTO_HARD_GROWTH_CAP: usize = 16_384;
const METRIC_WIDTH: usize = 11;
const METRIC_ROW_BYTES: u64 = 104;

struct DeviceFixture {
    _root: TempDir,
    root_path: PathBuf,
    settings: Settings,
    runtime_install: CanonicalNativeRuntimeInstallReceiptV1,
    happy_contract_sha256: String,
    stale_contract_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn eurusd_fixture(rows: usize) -> Ohlcv {
    let first_timestamp = 1_704_067_200_000_i64;
    let mut timestamp = Vec::with_capacity(rows);
    let mut open = Vec::with_capacity(rows);
    let mut high = Vec::with_capacity(rows);
    let mut low = Vec::with_capacity(rows);
    let mut close = Vec::with_capacity(rows);
    let mut volume = Vec::with_capacity(rows);
    for row in 0..rows {
        timestamp.push(first_timestamp + row as i64 * M15_MILLIS);
        let phase = row as f64;
        let row_open =
            1.08 + 0.0025 * (phase * 0.071).sin() + 0.0012 * (phase * 0.019).cos() + 2.0e-7 * phase;
        let row_close =
            row_open + 2.5e-4 * (phase * 0.37).sin() + ((row * 37 % 19) as f64 - 9.0) * 8.0e-6;
        open.push(row_open);
        close.push(row_close);
        high.push(row_open.max(row_close) + 7.0e-5 + (row % 3) as f64 * 1.0e-6);
        low.push(row_open.min(row_close) - 6.0e-5 - (row % 5) as f64 * 1.0e-6);
        volume.push(100.0 + (row % 23) as f64 * 3.0);
    }
    Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: Some(volume),
    }
}

fn aggregate_m30(base: &Ohlcv) -> Ohlcv {
    let timestamps = base.timestamp.as_deref().expect("base timestamps");
    let volume = base.volume.as_deref().expect("base volume");
    assert_eq!(base.len() % 2, 0, "M15 rows must aggregate exactly");
    let rows = base.len() / 2;
    let mut parent = Ohlcv {
        timestamp: Some(Vec::with_capacity(rows)),
        open: Vec::with_capacity(rows),
        high: Vec::with_capacity(rows),
        low: Vec::with_capacity(rows),
        close: Vec::with_capacity(rows),
        volume: Some(Vec::with_capacity(rows)),
    };
    for row in 0..rows {
        let first = row * 2;
        let second = first + 1;
        parent
            .timestamp
            .as_mut()
            .expect("parent timestamps")
            .push(timestamps[first]);
        parent.open.push(base.open[first]);
        parent.high.push(base.high[first].max(base.high[second]));
        parent.low.push(base.low[first].min(base.low[second]));
        parent.close.push(base.close[second]);
        parent
            .volume
            .as_mut()
            .expect("parent volume")
            .push(volume[first] + volume[second]);
    }
    parent
}

fn publish_fixture(
    root: &Path,
    namespace: &str,
    timeframe: CanonicalTimeframe,
    bars: &Ohlcv,
    expected_generation: Option<&str>,
    publication_tag: &str,
) -> Result<SelectedDatasetGenerationV1> {
    let identity = CanonicalDatasetIdentity::external(
        namespace,
        "EURUSD",
        timeframe,
        BarTimestampConvention::BarOpen,
    )
    .map_err(anyhow::Error::msg)
    .context("build fixture dataset identity")?;
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.search.canonical-native-discovery-device-fixture.v1",
        format!("{namespace}:{timeframe}:{publication_tag}").into_bytes(),
    )
    .context("seal fixture provenance")?;
    let published = publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity: &identity,
        expected_generation,
        provenance: &provenance,
        ohlcv: bars,
        volume: CanonicalVolumeRef::Float64(
            bars.volume.as_deref().expect("fixture volume is present"),
        ),
        rows_per_chunk: 256,
    })
    .context("publish immutable device fixture generation")?;
    SelectedDatasetGenerationV1::from_manifest(published.manifest())
        .context("select immutable device fixture generation")
}

fn device_settings(root: &Path) -> Settings {
    let mut settings = Settings::default();
    settings.system.data_dir = root.to_owned();
    settings.system.symbol = "EURUSD".to_owned();
    settings.system.account_currency = "USD".to_owned();
    settings.system.base_timeframe = "M15".to_owned();
    settings.system.higher_timeframes = vec!["M30".to_owned()];
    settings.system.multi_resolution_enabled = true;
    settings.models.prop_search_population = CONFIGURED_POPULATION;
    settings.models.prop_search_population_auto = true;
    settings.models.prop_search_generations = 0;
    settings.models.prop_search_max_indicators = MAX_INDICATORS;
    settings.models.prop_search_max_rows = 0;
    settings.models.prop_search_max_rows_by_tf.clear();
    settings.models.prop_search_device = "auto".to_owned();
    settings.models.prop_search_min_payoff_ratio = 0.0;
    settings.models.discovery_runtime.prefilter_top_k = 0;
    settings.models.discovery_runtime.min_history_years = 0;
    settings.models.discovery_runtime.adaptive_thresholds = false;
    settings.models.discovery_ledger.enabled = false;
    settings.models.gene_stop_bounds.atr_scaled = false;
    settings.models.search_runtime.seed = Some(0x5eed_2026_0827);
    settings.models.data_runtime.normalize_features = false;
    settings
}

fn fixture_contract(
    root: &Path,
    series: &CanonicalDatasetSeriesReceiptV1,
) -> Result<CanonicalTrendbarResearchExecutionContractV3> {
    let options = FeatureBuildOptions {
        higher_tfs: vec!["M30".to_owned()],
        ..FeatureBuildOptions::default()
    };
    let input = CanonicalSearchInput::from_exact_series_receipt(
        root,
        series,
        CanonicalTimeframe::M15,
        &options,
    )
    .context("build fixture-only CPU receipt over the exact immutable series")?;
    let input_receipt = input
        .receipt()
        .context("seal fixture-only canonical CPU input receipt")?;
    let assumption_payload = b"EURUSD/USD;pip=0.0001;pip-value=10;spread=1.2;slippage-per-fill=0.1;commission-per-fill=3.5;swap-long=-0.2;swap-short=-0.1;pnl-fee=0";
    let assumption_source_sha256 = sha256_hex(assumption_payload);
    CanonicalTrendbarResearchExecutionContractV3::new(
        input_receipt,
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "neoethos.test.eurusd-usd-financial-values.v1",
            assumption_source_sha256: &assumption_source_sha256,
            pip_size: 1.0e-4,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.2,
            slippage_pips_per_fill_assumption: 0.1,
            commission_account_per_lot_per_fill_assumption: 3.5,
            swap_long_pips_per_day: -0.2,
            swap_short_pips_per_day: -0.1,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .context("seal explicit fixture financial contract")
}

fn save_contract(
    root: &Path,
    relative_path: &str,
    contract: &CanonicalTrendbarResearchExecutionContractV3,
) -> Result<String> {
    let bytes = serde_json::to_vec(contract).context("serialize saved research contract")?;
    let path = root.join(relative_path);
    std::fs::create_dir_all(path.parent().context("contract parent")?)
        .context("create contract directory")?;
    std::fs::write(&path, &bytes).context("write saved research contract")?;
    Ok(sha256_hex(&bytes))
}

fn publish_series(
    root: &Path,
    namespace: &str,
    base: &Ohlcv,
    parent: &Ohlcv,
) -> Result<CanonicalDatasetSeriesReceiptV1> {
    let base_generation = publish_fixture(
        root,
        namespace,
        CanonicalTimeframe::M15,
        base,
        None,
        "base-v1",
    )?;
    let parent_generation = publish_fixture(
        root,
        namespace,
        CanonicalTimeframe::M30,
        parent,
        None,
        "parent-v1",
    )?;
    CanonicalDatasetSeriesReceiptV1::new(
        base_generation.clone(),
        vec![base_generation, parent_generation],
    )
    .context("seal exact M15/M30 fixture series")
}

fn setup_device_fixture() -> Result<DeviceFixture> {
    set_migration_enabled(false);
    let root = tempfile::tempdir().context("create exact device-fixture root")?;
    let root_path = root.path().to_path_buf();
    let base = eurusd_fixture(FIXTURE_ROWS);
    let parent = aggregate_m30(&base);

    let happy_series = publish_series(
        &root_path,
        "neoethos-canonical-native-device-happy",
        &base,
        &parent,
    )?;
    let happy_contract = fixture_contract(&root_path, &happy_series)?;
    let happy_contract_sha256 = save_contract(&root_path, HAPPY_CONTRACT, &happy_contract)?;

    let stale_namespace = "neoethos-canonical-native-device-stale";
    let stale_series = publish_series(&root_path, stale_namespace, &base, &parent)?;
    let stale_contract = fixture_contract(&root_path, &stale_series)?;
    let stale_contract_sha256 = save_contract(&root_path, STALE_CONTRACT, &stale_contract)?;
    let stale_base_generation = stale_series.anchor().generation_id().to_owned();
    let mut advanced_base = base.clone();
    advanced_base.close[0] += 1.0e-6;
    advanced_base.high[0] = advanced_base.high[0].max(advanced_base.close[0] + 1.0e-6);
    let advanced = publish_fixture(
        &root_path,
        stale_namespace,
        CanonicalTimeframe::M15,
        &advanced_base,
        Some(&stale_base_generation),
        "base-v2-after-contract",
    )?;
    assert_ne!(advanced.generation_id(), stale_base_generation);

    let settings = device_settings(&root_path);
    let runtime_install = install_and_seal_canonical_native_runtime_authority_v1(&settings)
        .context("install and seal canonical native runtime authority")?;
    Ok(DeviceFixture {
        _root: root,
        root_path,
        settings,
        runtime_install,
        happy_contract_sha256,
        stale_contract_sha256,
    })
}

fn contract_ref(
    relative_path: &str,
    expected_sha256: &str,
) -> Result<CanonicalResearchContractArtifactRefV1> {
    CanonicalResearchContractArtifactRefV1::checked_new(relative_path, expected_sha256)
        .map_err(anyhow::Error::new)
}

fn overrides() -> Result<CanonicalNativeGenerationZeroOverridesV1> {
    CanonicalNativeGenerationZeroOverridesV1::checked_new(
        Some(CONFIGURED_POPULATION),
        Some(true),
        Some(MAX_INDICATORS),
    )
    .map_err(anyhow::Error::new)
}

fn recompute_evidence_identity(bytes: &[u8], expected: &str) -> Result<String> {
    let text = std::str::from_utf8(bytes).context("result artifact is UTF-8")?;
    let suffix = format!(",\"evidence_identity_sha256\":\"{expected}\"}}");
    let identity_material = text
        .strip_suffix(&suffix)
        .context("evidence identity is the final compact-JSON field")?;
    let mut digest = Sha256::new();
    digest.update(RESULT_IDENTITY_DOMAIN_V1);
    digest.update(identity_material.as_bytes());
    digest.update(b"}");
    Ok(format!("{:x}", digest.finalize()))
}

fn assert_published_result(
    root: &Path,
    published: &PublishedCanonicalNativeGenerationZeroResearchV1,
) -> Result<()> {
    assert_eq!(published.engine(), "CudaNativeF64");
    assert_eq!(published.configured_population(), CONFIGURED_POPULATION);
    assert!(published.resolved_population() > CONFIGURED_POPULATION);
    assert_eq!(published.term_cap(), MAX_INDICATORS);
    assert!(published.resolved_population() <= published.population_cap());
    assert_eq!(
        published.hard_growth_cap(),
        published.population_cap().min(NATIVE_AUTO_HARD_GROWTH_CAP)
    );
    assert!(published.resolved_population() <= published.hard_growth_cap());
    assert!(published.stage1_row_end() > published.stage1_row_start());
    assert!(published.stage1_row_end() - published.stage1_row_start() >= 101);

    let population = published.resolved_population();
    assert_eq!(published.parent_h2d_bytes(), 0);
    assert_eq!(published.adaptive_h2d_bytes(), 0);
    assert_eq!(published.metric_rows(), population as u64);
    assert_eq!(
        published.metric_bytes(),
        (population as u64) * METRIC_ROW_BYTES
    );
    assert_eq!(published.gene_count(), population);
    assert_eq!(published.metric_row_count(), population);
    assert_eq!(published.metric_value_count_per_row(), METRIC_WIDTH);
    assert!(published.consumer_completion_confirmed());
    assert!(!published.replay_identity_sealed());

    for identity in [
        published.evidence_identity_sha256(),
        published.file_sha256(),
        published.financial_input_receipt_identity_sha256(),
        published.native_input_receipt_identity_sha256(),
        published.population_sizing_receipt_identity_sha256(),
    ] {
        assert!(is_lower_hex_sha256(identity), "invalid SHA-256: {identity}");
    }
    assert_ne!(
        published.financial_input_receipt_identity_sha256(),
        published.native_input_receipt_identity_sha256()
    );
    assert_ne!(
        published.financial_input_receipt_identity_sha256(),
        published.population_sizing_receipt_identity_sha256()
    );
    assert_ne!(
        published.native_input_receipt_identity_sha256(),
        published.population_sizing_receipt_identity_sha256()
    );

    let expected_relative_path = format!(
        "research/native-discovery/v1/cngr1-{}.json",
        published.evidence_identity_sha256()
    );
    assert_eq!(published.relative_path(), expected_relative_path);
    let artifact = std::fs::read(root.join(published.relative_path()))
        .context("reopen content-addressed Generation-zero result")?;
    assert_eq!(published.byte_count(), artifact.len() as u64);
    assert_eq!(published.file_sha256(), sha256_hex(&artifact));
    assert_eq!(
        recompute_evidence_identity(&artifact, published.evidence_identity_sha256())?,
        published.evidence_identity_sha256()
    );

    let wire: Value = serde_json::from_slice(&artifact).context("decode result artifact JSON")?;
    assert_eq!(
        wire["schema"].as_str(),
        Some(CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1)
    );
    assert_eq!(
        wire["version"].as_u64(),
        Some(u64::from(
            CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1
        ))
    );
    assert_eq!(
        wire["evidence_identity_sha256"].as_str(),
        Some(published.evidence_identity_sha256())
    );

    let sizing = &wire["population_sizing"];
    assert_eq!(
        wire["financial_provenance_only"]["cpu_receipt_id"].as_str(),
        Some(published.financial_input_receipt_identity_sha256())
    );
    assert_eq!(
        wire["evaluated_native_input"]["receipt_id"].as_str(),
        Some(published.native_input_receipt_identity_sha256())
    );
    assert_eq!(
        sizing["receipt_id"].as_str(),
        Some(published.population_sizing_receipt_identity_sha256())
    );
    assert!(
        sizing["prepared_feature_count"]
            .as_u64()
            .is_some_and(|count| count >= MAX_INDICATORS as u64)
    );
    assert_eq!(
        sizing["raw_configured_max_indicators"].as_u64(),
        Some(MAX_INDICATORS as u64)
    );
    assert_eq!(
        sizing["resolved_max_indicators"].as_u64(),
        Some(MAX_INDICATORS as u64)
    );
    assert_eq!(
        sizing["configured_population"].as_u64(),
        Some(CONFIGURED_POPULATION as u64)
    );
    assert_eq!(
        sizing["resolved_population"].as_u64(),
        Some(population as u64)
    );
    assert_eq!(
        sizing["population_cap"].as_u64(),
        Some(published.population_cap() as u64)
    );
    assert_eq!(
        sizing["hard_growth_cap"].as_u64(),
        Some(published.hard_growth_cap() as u64)
    );
    assert_eq!(sizing["term_cap"].as_u64(), Some(MAX_INDICATORS as u64));
    assert_eq!(
        sizing["stage1_row_start"].as_u64(),
        Some(published.stage1_row_start() as u64)
    );
    assert_eq!(
        sizing["stage1_row_end"].as_u64(),
        Some(published.stage1_row_end() as u64)
    );
    assert_eq!(
        sizing["selected_device_ordinal"].as_u64(),
        Some(u64::from(published.selected_device_ordinal()))
    );

    let genes = wire["generation_zero_evaluation"]["genes"]
        .as_array()
        .context("result genes are an array")?;
    let metrics = wire["generation_zero_evaluation"]["metrics"]
        .as_array()
        .context("result metrics are an array")?;
    assert_eq!(genes.len(), population);
    assert_eq!(metrics.len(), population);
    assert!(
        genes
            .iter()
            .all(|gene| gene["generation"].as_u64() == Some(0))
    );
    assert!(metrics.iter().all(|row| {
        row.as_array().is_some_and(|values| {
            values.len() == METRIC_WIDTH
                && values
                    .iter()
                    .all(|value| value.as_f64().is_some_and(f64::is_finite))
        })
    }));

    let counters = &wire["residency_counters"];
    assert_eq!(counters["parent_upload_count"].as_u64(), Some(0));
    assert_eq!(counters["parent_upload_bytes"].as_u64(), Some(0));
    assert_eq!(counters["adaptive_upload_bytes"].as_u64(), Some(0));
    assert_eq!(
        counters["metric_rows_readback_rows"].as_u64(),
        Some(population as u64)
    );
    assert_eq!(
        counters["metric_rows_readback_bytes"].as_u64(),
        Some((population as u64) * METRIC_ROW_BYTES)
    );
    assert_eq!(wire["completion"]["engine"].as_str(), Some("CudaNativeF64"));
    assert_eq!(
        wire["completion"]["consumer_completion_confirmed"].as_bool(),
        Some(true)
    );
    assert_eq!(
        wire["replay"]["replay_identity_sealed"].as_bool(),
        Some(false)
    );
    Ok(())
}

#[test]
fn real_rtx_public_native_discovery_entry_cancels_safely_and_publishes_exact_result() -> Result<()>
{
    if std::env::var(RUN_ENV).as_deref() != Ok("1") {
        eprintln!("SKIP real canonical native discovery entry; set {RUN_ENV}=1");
        return Ok(());
    }

    // Everything above this boundary is immutable fixture/contract setup.
    // Execution below reaches Search only through the public from-reference API.
    let fixture = setup_device_fixture()?;

    let cancelled_before_load = CanonicalNativeCancellationTokenV1::new();
    cancelled_before_load.cancel();
    let mut cancelled_before_load_progress = Vec::new();
    let cancelled_before_load_error = run_canonical_native_discovery_generation_zero_from_ref_v1(
        &fixture.settings,
        &fixture.runtime_install,
        contract_ref(
            "research/contracts/cancelled-before-load.json",
            &"1".repeat(64),
        )?,
        overrides()?,
        &cancelled_before_load,
        |event| cancelled_before_load_progress.push(event),
    )
    .expect_err("pre-cancelled execution must stop before artifact load");
    assert_eq!(
        cancelled_before_load_error.stage(),
        CanonicalNativeDiscoveryExecutionStageV1::ContractArtifactRead
    );
    assert_eq!(
        cancelled_before_load_error.code(),
        CanonicalNativeDiscoveryExecutionErrorCodeV1::Cancelled
    );
    assert!(cancelled_before_load_progress.is_empty());

    let cancelled_before_launch = CanonicalNativeCancellationTokenV1::new();
    let cancellation_from_progress = cancelled_before_launch.clone();
    let mut cancelled_before_launch_progress = Vec::new();
    let cancelled_before_launch_error = run_canonical_native_discovery_generation_zero_from_ref_v1(
        &fixture.settings,
        &fixture.runtime_install,
        contract_ref(HAPPY_CONTRACT, &fixture.happy_contract_sha256)?,
        overrides()?,
        &cancelled_before_launch,
        |event| {
            if matches!(event, DiscoveryProgress::SearchStarted { .. }) {
                cancellation_from_progress.cancel();
            }
            cancelled_before_launch_progress.push(event);
        },
    )
    .expect_err("cancellation from SearchStarted must stop at the final pre-launch gate");
    assert_eq!(
        cancelled_before_launch_error.stage(),
        CanonicalNativeDiscoveryExecutionStageV1::GenerationZeroEvaluation
    );
    assert_eq!(
        cancelled_before_launch_error.code(),
        CanonicalNativeDiscoveryExecutionErrorCodeV1::Cancelled
    );
    assert!(cancelled_before_launch.is_cancelled());
    assert!(
        !cancelled_before_launch_progress
            .iter()
            .any(|event| matches!(
                event,
                DiscoveryProgress::StageAdvanced { stage, .. }
                    if *stage == "resident_cuda_generation_zero"
            ))
    );
    assert!(
        !fixture
            .root_path
            .join("research/native-discovery/v1")
            .exists()
    );

    let cancelled_during_launch = CanonicalNativeCancellationTokenV1::new();
    let cancellation_after_gate = cancelled_during_launch.clone();
    let mut cancellation_thread = None;
    let mut cancelled_during_launch_progress = Vec::new();
    let cancelled_during_launch_error = run_canonical_native_discovery_generation_zero_from_ref_v1(
        &fixture.settings,
        &fixture.runtime_install,
        contract_ref(HAPPY_CONTRACT, &fixture.happy_contract_sha256)?,
        overrides()?,
        &cancelled_during_launch,
        |event| {
            if matches!(event, DiscoveryProgress::SearchStarted { .. }) {
                let token = cancellation_after_gate.clone();
                cancellation_thread = Some(std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    token.cancel();
                }));
            }
            cancelled_during_launch_progress.push(event);
        },
    )
    .expect_err("in-flight cancellation must wait for completion and skip publication");
    cancellation_thread
        .expect("SearchStarted spawned the cancellation worker")
        .join()
        .expect("cancellation worker completed");
    assert_eq!(
        cancelled_during_launch_error.stage(),
        CanonicalNativeDiscoveryExecutionStageV1::ResultPublication
    );
    assert_eq!(
        cancelled_during_launch_error.code(),
        CanonicalNativeDiscoveryExecutionErrorCodeV1::Cancelled
    );
    assert!(
        cancelled_during_launch_progress
            .iter()
            .any(|event| matches!(
                event,
                DiscoveryProgress::StageAdvanced { stage, .. }
                    if *stage == "resident_cuda_generation_zero"
            ))
    );
    assert!(
        !fixture
            .root_path
            .join("research/native-discovery/v1")
            .exists()
    );

    let cancellation = CanonicalNativeCancellationTokenV1::new();

    let mut first_progress = Vec::new();
    let first = run_canonical_native_discovery_generation_zero_from_ref_v1(
        &fixture.settings,
        &fixture.runtime_install,
        contract_ref(HAPPY_CONTRACT, &fixture.happy_contract_sha256)?,
        overrides()?,
        &cancellation,
        |event| first_progress.push(event),
    )
    .context("execute the first public canonical native Generation-zero run")?;
    assert!(!first.reused_identical());
    assert_published_result(&fixture.root_path, &first)?;
    assert!(first_progress.iter().any(|event| matches!(
        event,
        DiscoveryProgress::SearchStarted {
            population,
            generations: 0,
            max_indicators
        } if *population == first.resolved_population()
            && *max_indicators == MAX_INDICATORS
    )));
    assert!(first_progress.iter().any(|event| matches!(
        event,
        DiscoveryProgress::StageAdvanced { stage, .. }
            if *stage == "resident_cuda_generation_zero"
    )));

    let mut stale_progress = Vec::new();
    let stale_error = run_canonical_native_discovery_generation_zero_from_ref_v1(
        &fixture.settings,
        &fixture.runtime_install,
        contract_ref(STALE_CONTRACT, &fixture.stale_contract_sha256)?,
        overrides()?,
        &cancellation,
        |event| stale_progress.push(event),
    )
    .expect_err("an advance after contract sealing must fail closed");
    assert_eq!(
        stale_error.stage(),
        CanonicalNativeDiscoveryExecutionStageV1::ExactSourcePin
    );
    assert_eq!(
        stale_error.code(),
        CanonicalNativeDiscoveryExecutionErrorCodeV1::ExactGenerationConflict
    );
    assert!(
        stale_progress.is_empty(),
        "the stale exact generation reached native execution progress"
    );

    eprintln!(
        "CANONICAL_NATIVE_DISCOVERY_V1_DEVICE_EVIDENCE engine={} ordinal={} configured_population={} resolved_population={} population_cap={} hard_growth_cap={} term_cap={} stage1={}..{} parent_h2d_bytes={} adaptive_h2d_bytes={} metric_rows={} metric_bytes={} genes={} metric_width={} completion_confirmed={} replay_identity_sealed={} result={} file_sha256={} reused_identical={}",
        first.engine(),
        first.selected_device_ordinal(),
        first.configured_population(),
        first.resolved_population(),
        first.population_cap(),
        first.hard_growth_cap(),
        first.term_cap(),
        first.stage1_row_start(),
        first.stage1_row_end(),
        first.parent_h2d_bytes(),
        first.adaptive_h2d_bytes(),
        first.metric_rows(),
        first.metric_bytes(),
        first.gene_count(),
        first.metric_value_count_per_row(),
        first.consumer_completion_confirmed(),
        first.replay_identity_sealed(),
        first.relative_path(),
        first.file_sha256(),
        first.reused_identical(),
    );

    Ok(())
}
