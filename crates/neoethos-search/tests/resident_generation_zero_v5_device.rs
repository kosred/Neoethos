#![cfg(all(feature = "gpu-cuda", target_os = "linux"))]

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};
use neoethos_core::Settings;
use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::core::features::{FeatureBuildOptions, FeatureProfile};
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1,
    CanonicalOhlcvPublishRequest, CanonicalTimeframe, CanonicalVolumeRef, Ohlcv,
    PreparedGpuOnlyFeatureMaterializationV3, SelectedDatasetGenerationV1,
    install_data_runtime_overrides,
    materialize_prepared_gpu_only_feature_store_for_data_population_v3,
    pin_exact_canonical_series_v1, preflight_gpu_only_feature_workspace_v3,
    prepare_gpu_only_feature_materialization_v3, publish_canonical_ohlcv_generation,
};
use neoethos_search::data_selection::CanonicalSearchInput;
use neoethos_search::{
    CanonicalGpuResidentSearchInputReceiptV3, CanonicalTrendbarResearchCostAssumptionsV2,
    CanonicalTrendbarResearchExecutionContractV3, DiscoveryConfig, DiscoveryProgress,
    install_search_runtime_overrides_from_settings,
    prepare_staged_canonical_trendbar_research_run_input_v5,
    run_prepared_canonical_trendbar_research_generation_zero_v5, set_migration_enabled,
};
use sha2::{Digest, Sha256};

const M15_MILLIS: i64 = 15 * 60 * 1_000;
const FIXTURE_ROWS: usize = 1_200;
const CONFIGURED_POPULATION: usize = 200;
const MAX_INDICATORS: usize = 5;

fn hex32(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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
        "neoethos.search.resident-generation-zero-v5-device-fixture.v1",
        format!("{namespace}:{timeframe}").into_bytes(),
    )
    .context("seal fixture provenance")?;
    let published = publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity: &identity,
        expected_generation: None,
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

fn prepare_gpu_recipe(
    root: &Path,
    series: CanonicalDatasetSeriesReceiptV1,
) -> Result<PreparedGpuOnlyFeatureMaterializationV3> {
    let pinned =
        pin_exact_canonical_series_v1(root, series).context("pin exact M15/M30 device fixture")?;
    let preflight = preflight_gpu_only_feature_workspace_v3(
        pinned,
        CanonicalTimeframe::M15,
        FeatureProfile::Standard,
        FIXTURE_ROWS,
    )
    .context("preflight exact GPU-only feature recipe")?;
    prepare_gpu_only_feature_materialization_v3(preflight)
        .context("prepare exact GPU-only feature materialization")
}

fn device_settings() -> Settings {
    let mut settings = Settings::default();
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
    // The legacy CPU V2 receipt is fixture-only financial/source authority.
    // The V5 dispatcher below is independently native-only and admits the
    // resident GPU route explicitly; keeping this legacy selector at `auto`
    // prevents it from falsely demanding an incomplete full GPU feature graph.
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
    let assumption_source_sha256 = format!("{:x}", Sha256::digest(assumption_payload));
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

#[test]
fn real_rtx_v5_generation_zero_uses_admitted_resident_cuda_without_parent_h2d() -> Result<()> {
    if std::env::var("NEOETHOS_RUN_RESIDENT_GEN0_V5_DEVICE_TEST").as_deref() != Ok("1") {
        eprintln!("SKIP real RTX V5 Generation-0; set NEOETHOS_RUN_RESIDENT_GEN0_V5_DEVICE_TEST=1");
        return Ok(());
    }

    let settings = device_settings();
    install_data_runtime_overrides(settings.models.data_runtime.normalize_features);
    install_search_runtime_overrides_from_settings(&settings);
    set_migration_enabled(false);

    let root = tempfile::tempdir().context("create exact device-fixture root")?;
    let base = eurusd_fixture(FIXTURE_ROWS);
    let parent = aggregate_m30(&base);
    let namespace = "neoethos-search-v5-generation-zero-device";
    let base_generation = publish_fixture(root.path(), namespace, CanonicalTimeframe::M15, &base)?;
    let parent_generation =
        publish_fixture(root.path(), namespace, CanonicalTimeframe::M30, &parent)?;
    let series = CanonicalDatasetSeriesReceiptV1::new(
        base_generation.clone(),
        vec![base_generation, parent_generation],
    )
    .context("seal exact M15/M30 fixture series")?;
    let anchor = series.anchor().identity().clone();
    let contract = fixture_contract(root.path(), &series)?;
    let mut config =
        DiscoveryConfig::try_from_settings_for_canonical_trendbar_research(&settings, &contract)?;
    config.session_spread_pips = None;
    config.cost_band_pips = None;

    // Exercise the real prepared-recipe extent guard. Data recipe preflight is
    // allowed, but no admitted store may materialize when canonical prefilter
    // semantics would change the resident feature universe.
    let materialized_bad_route = Arc::new(AtomicBool::new(false));
    let materialized_bad_route_for_factory = Arc::clone(&materialized_bad_route);
    let mut reducing_config = config.clone();
    reducing_config.runtime_overrides.prefilter_top_k = 1;
    let bad_root = root.path().to_path_buf();
    let bad_series = series.clone();
    let rejected = prepare_staged_canonical_trendbar_research_run_input_v5(
        &reducing_config,
        &contract,
        move |_native_facts| prepare_gpu_recipe(&bad_root, bad_series),
        move |_prepared, _admitted| {
            materialized_bad_route_for_factory.store(true, Ordering::SeqCst);
            Err(anyhow!("guard failure reached native materialization"))
        },
    )
    .expect_err("a reducing canonical prefilter must fail before Data allocation");
    assert!(
        format!("{rejected:#}").contains("requires resident feature prefilter/remap before sizing"),
        "unexpected pre-allocation rejection: {rejected:#}"
    );
    assert!(
        !materialized_bad_route.load(Ordering::SeqCst),
        "the reducing prefilter reached native Data materialization"
    );

    let actual_root = root.path().to_path_buf();
    let actual_series = series.clone();
    let actual_anchor = anchor.clone();
    let prepared = prepare_staged_canonical_trendbar_research_run_input_v5(
        &config,
        &contract,
        move |_native_facts| prepare_gpu_recipe(&actual_root, actual_series),
        move |prepared, admitted| {
            let store = materialize_prepared_gpu_only_feature_store_for_data_population_v3(
                prepared, admitted,
            )
            .context("materialize admitted Data+population resident store")?;
            assert_eq!(
                hex32(store.final_feature_plan_v3_sha256()),
                store.feature_plan().identity().to_hex(),
                "final resident feature plan identity detached from its sealed recipe"
            );
            assert_eq!(
                hex32(store.source_provenance_sha256()),
                store.source_provenance().identity().to_hex(),
                "resident source-provenance identity detached from its sealed recipe"
            );
            let receipt = CanonicalGpuResidentSearchInputReceiptV3::from_resident_store(
                &actual_anchor,
                &store,
            )
            .context("seal native GPU-resident Search receipt")?;
            Ok((receipt, store))
        },
    )
    .context("prepare exact V5 Data+resident-population run")?;

    let sizing = prepared.population_sizing_receipt_v2();
    assert!(sizing.population_auto());
    assert_eq!(sizing.configured_population(), CONFIGURED_POPULATION);
    assert!(
        sizing.resolved_population() > CONFIGURED_POPULATION,
        "population_auto did not grow the configured population"
    );
    assert_eq!(sizing.term_cap(), MAX_INDICATORS);
    assert!(sizing.adaptive_base_effective_for_stage1());
    assert!(sizing.stage1_row_end() - sizing.stage1_row_start() >= 101);
    let expected_population = sizing.resolved_population();
    let expected_term_cap = sizing.term_cap();
    let expected_launches = expected_population.div_ceil(sizing.max_concurrent_scenario_count());
    let expected_stage1 = (sizing.stage1_row_start(), sizing.stage1_row_end());

    let mut progress = Vec::new();
    let milestone =
        run_prepared_canonical_trendbar_research_generation_zero_v5(prepared, |event| {
            progress.push(event)
        })
        .context("execute real resident-CUDA Generation-0")?;

    assert_eq!(milestone.engine(), "CudaNativeF64");
    assert_eq!(milestone.resolved_population(), expected_population);
    assert_eq!(milestone.term_cap(), expected_term_cap);
    assert_eq!(
        (milestone.stage1_row_start(), milestone.stage1_row_end()),
        expected_stage1
    );
    assert_eq!(
        milestone.metrics_receipt_identities_sha256().len(),
        expected_launches
    );
    assert!(
        milestone
            .metrics_receipt_identities_sha256()
            .iter()
            .all(|identity| *identity != [0; 32])
    );
    assert!(milestone.adaptive_token_identity_sha256().is_some());
    assert!(milestone.consumer_completion_confirmed());
    assert!(!milestone.replay_identity_sealed());
    assert_eq!(milestone.search_result().genes.len(), expected_population);
    assert_eq!(milestone.search_result().metrics.len(), expected_population);
    assert!(progress.iter().any(|event| matches!(
        event,
        DiscoveryProgress::SearchStarted { population, generations: 0, max_indicators }
            if *population == expected_population && *max_indicators == MAX_INDICATORS
    )));
    assert!(progress.iter().any(|event| matches!(
        event,
        DiscoveryProgress::StageAdvanced { stage, .. }
            if *stage == "resident_cuda_generation_zero"
    )));

    let counters = milestone.residency_counters();
    assert_eq!(counters.parent_upload_count(), 0);
    assert_eq!(counters.parent_upload_bytes(), 0);
    assert_eq!(counters.stream_creation_count(), 0);
    assert_eq!(counters.adaptive_upload_bytes(), 0);
    assert_eq!(counters.view_binding_count(), 1);
    assert_eq!(counters.ordered_binding_count(), 0);
    assert_eq!(
        counters.metric_rows_readback_count(),
        u64::try_from(expected_launches).expect("launch count")
    );
    assert_eq!(
        counters.metric_rows_readback_rows(),
        u64::try_from(expected_population).expect("population rows")
    );
    assert_eq!(
        counters.metric_rows_readback_bytes(),
        u64::try_from(expected_population)
            .expect("population bytes")
            .checked_mul(104)
            .expect("metric bytes")
    );

    eprintln!(
        "V5_GEN0_RTX_EVIDENCE engine={} ordinal={} configured_population={} resolved_population={} term_cap={} stage1={}..{} launches={} native_receipt={} sizing_receipt={} adaptive_token={} parent_h2d_bytes={} adaptive_h2d_bytes={} metric_rows={} metric_bytes={} completion_confirmed={} replay_identity_sealed={}",
        milestone.engine(),
        milestone.selected_device_ordinal(),
        CONFIGURED_POPULATION,
        milestone.resolved_population(),
        milestone.term_cap(),
        milestone.stage1_row_start(),
        milestone.stage1_row_end(),
        milestone.metrics_receipt_identities_sha256().len(),
        milestone.native_input_receipt_identity_sha256(),
        milestone.population_sizing_receipt_identity_sha256(),
        milestone
            .adaptive_token_identity_sha256()
            .map(|identity| format!("{identity:02x?}"))
            .unwrap_or_else(|| "none".to_owned()),
        counters.parent_upload_bytes(),
        counters.adaptive_upload_bytes(),
        counters.metric_rows_readback_rows(),
        counters.metric_rows_readback_bytes(),
        milestone.consumer_completion_confirmed(),
        milestone.replay_identity_sealed(),
    );

    Ok(())
}
