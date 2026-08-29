use super::{
    RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3, ResidentFeatureColumnBindingV3,
    ResidentFeatureStoreCudaErrorV3, ResidentFeatureStoreSearchStartErrorV2,
    ResidentPopulationSessionV3,
};
use crate::full_discovery_workspace_plan_v1::seal_test_full_discovery_run_device_v3;
use crate::population::{
    CudaPopulationError, PopulationEvaluationViewV1, PopulationGeneView, PopulationTimestampModeV1,
    ResidentAdaptiveBaseRequestV1, STATUS_ADAPTIVE_BASE_DEGENERATE,
    STATUS_STRICT_RESIDENT_POISONED,
};
use crate::resident_generation_v1::{
    ParentSelectionPolicyV1, ResidentGenerationPlanAuthorityInputV1,
    SealedResidentGenerationPlanV1, SurvivorSelectionPolicyV1,
    discovery_generation_semantics_sha256_v1, seal_resident_generation_plan_v1,
};
#[cfg(feature = "cuda-device-fixtures")]
use crate::resident_search_v2::resident_search_v2_production_readiness;
use crate::resident_smc_v3::{
    RESIDENT_SMC_COLUMN_NAMES_V3, begin_resident_smc_store_v3, prepare_resident_smc_parent_v3,
};
use crate::{
    GeneDescriptor, NeoPopulationSettings, SMC_SLOTS, ScenarioDescriptor,
    acquire_discovery_run_device_admission_v1,
};
use neoethos_gpu_contracts::ABI_VERSION;
use neoethos_gpu_contracts::resident_feature_store_v3::ResidentWorkingSetRequestV3;
#[cfg(feature = "cuda-device-fixtures")]
use sha2::{Digest, Sha256};
use std::sync::Arc;

const ROWS: usize = 160;
const CANDIDATE_ID: u64 = 7;
const SCENARIO_ID: u64 = 11;

fn fixture_search_plan() -> SealedResidentGenerationPlanV1 {
    seal_resident_generation_plan_v1(ResidentGenerationPlanAuthorityInputV1 {
        parent_selection: ParentSelectionPolicyV1::RankWeighted,
        survivor_selection: SurvivorSelectionPolicyV1::RankWeighted,
        max_terms_per_gene: 3,
        minimum_terms_per_gene: 1,
        logical_population_count: 1,
        retained_evaluation_capacity: 1,
        feature_count: RESIDENT_SMC_COLUMN_NAMES_V3.len(),
        generation_count: 1,
        survivor_count: 1,
        immigrant_count: 0,
        search_seed: 0x9d2c_a877_61e4_05b3,
        mutation_intensity_q32: 0,
        threshold_ladder_bits: std::array::from_fn(|index| {
            (0.05_f64 * (index as f64 + 1.0)).to_bits()
        }),
        stop_bounds_bits: std::array::from_fn(|index| (index as f64 + 1.0).to_bits()),
        smc_probability_q32: [0; SMC_SLOTS],
        generation_semantics_sha256: discovery_generation_semantics_sha256_v1(),
        run_identity_sha256: [0x71; 32],
        strategy_gene_schema_sha256: [0x72; 32],
        rank_semantics_sha256: [0x73; 32],
        metric_semantics_sha256: [0x74; 32],
        scoring_semantics_sha256: [0x75; 32],
        novelty_semantics_sha256: [0x76; 32],
        scenario_order_semantics_sha256: [0x77; 32],
        cuda_build_manifest_sha256: [0x78; 32],
        rng_mapping_sha256: [0x79; 32],
    })
    .expect("fixture Search plan is sealed")
}

fn fixture_bindings() -> Vec<ResidentFeatureColumnBindingV3> {
    RESIDENT_SMC_COLUMN_NAMES_V3
        .iter()
        .enumerate()
        .map(|(ordinal, name)| ResidentFeatureColumnBindingV3 {
            ordinal,
            feature_name: (*name).to_owned(),
            canonical_parameter_tuple_sha256: [(ordinal + 1) as u8; 32],
            route_receipt_sha256: [(ordinal + 65) as u8; 32],
        })
        .collect()
}

fn fixture_ohlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<i64>) {
    let mut open = Vec::with_capacity(ROWS);
    let mut high = Vec::with_capacity(ROWS);
    let mut low = Vec::with_capacity(ROWS);
    let mut close = Vec::with_capacity(ROWS);
    let mut volume = Vec::with_capacity(ROWS);
    let mut timestamps = Vec::with_capacity(ROWS);
    for row in 0..ROWS {
        let base = 1.08 + row as f64 * 0.000_01;
        let delta = match row % 4 {
            0 => 0.000_03,
            1 => -0.000_02,
            2 => 0.000_01,
            _ => -0.000_04,
        };
        let row_close = base + delta;
        open.push(base);
        high.push(base.max(row_close) + 0.000_07 + (row % 3) as f64 * 0.000_001);
        low.push(base.min(row_close) - 0.000_06 - (row % 5) as f64 * 0.000_001);
        close.push(row_close);
        volume.push(1_000.0 + row as f64 * 0.25);
        timestamps.push(1_704_067_200_000 + row as i64 * 300_000);
    }
    (open, high, low, close, volume, timestamps)
}

fn exact_working_set(
    run_device: &super::GpuOnlyRunDeviceAdmissionV3,
    bindings: &[ResidentFeatureColumnBindingV3],
    retained_feature_device_bytes: usize,
) -> Result<
    neoethos_gpu_contracts::resident_feature_store_v3::ResidentWorkingSetBoundV3,
    Box<dyn std::error::Error>,
> {
    let pointer_table_bytes = bindings
        .len()
        .checked_mul(4 * std::mem::size_of::<u64>())
        .ok_or("pointer table byte overflow")?;
    let name_offset_bytes = bindings
        .len()
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
        .ok_or("name offset byte overflow")?;
    let name_bytes = bindings.iter().try_fold(0_usize, |sum, binding| {
        sum.checked_add(binding.feature_name.len())
            .ok_or("feature name byte overflow")
    })?;
    Ok(ResidentWorkingSetRequestV3 {
        row_count: ROWS,
        column_count: bindings.len(),
        max_live_producer_bytes: retained_feature_device_bytes as u64,
        max_live_producer_scratch_bytes: 0,
        normalization_scratch_bytes: 0,
        fit_metadata_bytes: 0,
        pointer_and_schema_metadata_bytes: pointer_table_bytes
            .checked_add(name_offset_bytes)
            .and_then(|bytes| bytes.checked_add(name_bytes))
            .ok_or("pointer/schema metadata byte overflow")?
            as u64,
        device_free_bytes_snapshot: run_device.phase_one_free_bytes_snapshot(),
        allocator_context_reserve_bytes: run_device.allocator_context_reserve_bytes(),
        reserve_policy_id: RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3.to_owned(),
    }
    .seal()?)
}

fn wait_for_batch_retirement(
    assembler: &mut super::ResidentFeatureStoreAssemblerV3,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    while !assembler.try_retire_completed_batch()? {
        std::thread::yield_now();
    }
    Ok(())
}

fn adaptive_failure_session(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps: &[i64],
) -> Result<ResidentPopulationSessionV3, Box<dyn std::error::Error>> {
    let admission = acquire_discovery_run_device_admission_v1()?;
    let run_device = seal_test_full_discovery_run_device_v3(admission, 4 * 1024 * 1024, 1024)?;
    let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
    let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
    let ordinal = run_device.device_identity().ordinal();
    let bindings = fixture_bindings();
    let materialization = prepare_resident_smc_parent_v3(
        &run_device,
        open,
        high,
        low,
        close,
        volume,
        timestamps,
        bindings.clone(),
    )?;
    let working_set = exact_working_set(
        &run_device,
        &bindings,
        materialization.receipt().retained_feature_device_bytes,
    )?;
    let (mut assembler, pending) =
        begin_resident_smc_store_v3(run_device, bindings, &working_set, materialization)?;
    pending.append_to(&mut assembler)?;
    wait_for_batch_retirement(&mut assembler)?;
    let owner = assembler.seal()?;
    loop {
        match owner.compact_hashes_if_ready() {
            Ok(_) => break,
            Err(ResidentFeatureStoreCudaErrorV3::NotReady) => std::thread::yield_now(),
            Err(error) => return Err(error.into()),
        }
    }
    let resident_import = owner.import_on_consumer_stream(context, stream, ordinal)?;
    Ok(resident_import.consume_into_population_session_v3()?)
}

fn assert_adaptive_evaluation_fails_closed(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps: &[i64],
    pip_size: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut session = adaptive_failure_session(open, high, low, close, volume, timestamps)?;
    let view = PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let request = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&view, pip_size, 1, 0)?;
    session.bind_evaluation_view_with_resident_adaptive_base_v1(view, request)?;

    let descriptors = [GeneDescriptor {
        candidate_id: CANDIDATE_ID,
        term_offset: 0,
        term_count: 1,
        long_threshold: 1.0e300,
        short_threshold: -1.0e300,
        stop_ticks: 100,
        target_ticks: 200,
        stop_vol_multiplier: 1.0,
        flags: 0,
        reserved: 0,
    }];
    let offsets = [0_i32, 1];
    let indices = [0_i32];
    let weights = [0.0_f64];
    let stop_pips = [10.0_f64];
    let target_pips = [20.0_f64];
    let stop_vol_multipliers = [1.0_f64];
    let smc_flags = [0_i8; SMC_SLOTS];
    let smc_weights = [0.0_f64; SMC_SLOTS];
    session.upload_genes(PopulationGeneView {
        descriptors: &descriptors,
        offsets: &offsets,
        indices: &indices,
        weights: &weights,
        stop_pips: &stop_pips,
        target_pips: &target_pips,
        stop_vol_multipliers: &stop_vol_multipliers,
        smc_flags: &smc_flags,
        smc_weights: &smc_weights,
        gate_threshold: 1.0e300,
        smc_gate_disabled: true,
    })?;
    session.upload_scenarios(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ROWS as u32,
        ..ScenarioDescriptor::default()
    }])?;
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        max_hold_bars: 8,
        min_hold_bars: 1,
        max_trades_per_day: 1,
        month_capacity: 12,
        gap_threshold_ms: 600_000,
        initial_equity: 100_000.0,
        pip_value: 0.000_1,
        spread_pips: 1.0,
        commission_per_trade: 7.0,
        pip_value_per_lot: 10.0,
        risk_per_trade_min: 0.005,
        risk_per_trade_max: 0.01,
        high_quality_confidence: 0.75,
        spread_pips_asian: 1.0,
        spread_pips_overlap: 1.0,
        spread_pips_late_ny: 1.0,
        ..NeoPopulationSettings::default()
    };
    let result = session
        .enqueue_metrics_only_v1(&settings)?
        .consume_terminal_compact_result_v1();
    match result {
        Err(CudaPopulationError::Native { status, .. }) => {
            assert_eq!(status, STATUS_ADAPTIVE_BASE_DEGENERATE);
            Ok(())
        }
        Err(error) => Err(format!("unexpected adaptive failure: {error}").into()),
        Ok(_) => Err("degenerate resident adaptive base silently produced metrics".into()),
    }
}

#[test]
fn resident_adaptive_constant_candles_fail_closed_before_metrics_are_accepted()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }
    let open = vec![1.08_f64; ROWS];
    let high = vec![1.08_f64; ROWS];
    let low = vec![1.08_f64; ROWS];
    let close = vec![1.08_f64; ROWS];
    let volume = vec![1_000.0_f64; ROWS];
    let timestamps = (0..ROWS)
        .map(|row| 1_704_067_200_000 + row as i64 * 300_000)
        .collect::<Vec<_>>();
    assert_adaptive_evaluation_fails_closed(
        &open,
        &high,
        &low,
        &close,
        &volume,
        &timestamps,
        0.000_1,
    )
}

#[test]
fn resident_adaptive_tiny_positive_pip_overflow_fails_closed_before_metrics_are_accepted()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    assert_adaptive_evaluation_fails_closed(
        &open,
        &high,
        &low,
        &close,
        &volume,
        &timestamps,
        f64::from_bits(1),
    )
}

#[test]
fn resident_adaptive_checked_bind_validates_the_current_token_and_poison_rejects_uploads()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();

    let mut session_a = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    let view_a =
        PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let request_a = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&view_a, 0.000_1, 1, 0)?;
    let rejected = session_a.bind_evaluation_view_with_resident_adaptive_base_checked_v1(
        view_a,
        request_a,
        |_current_token| {
            Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "receipt rejected the exact current token".into(),
            ))
        },
    );
    assert!(matches!(
        rejected,
        Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(message))
            if message == "receipt rejected the exact current token"
    ));
    let rejected_upload = session_a.upload_scenarios(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ROWS as u32,
        ..ScenarioDescriptor::default()
    }]);
    assert!(matches!(
        rejected_upload,
        Err(ResidentFeatureStoreCudaErrorV3::Population(
            CudaPopulationError::Native {
                status: STATUS_STRICT_RESIDENT_POISONED,
                ..
            }
        ))
    ));
    let lease_a = session_a.record_consumer_completion()?;
    while !lease_a.completion_is_ready()? {
        std::thread::yield_now();
    }
    drop(lease_a);

    let mut session_b = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    let view_b =
        PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let request_b = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&view_b, 0.000_1, 1, 0)?;
    let mut validator_calls = 0_u32;
    let accepted = session_b.bind_evaluation_view_with_resident_adaptive_base_checked_v1(
        view_b,
        request_b,
        |current_token| {
            validator_calls += 1;
            assert_eq!(
                current_token.request_identity_sha256(),
                request_b.identity_sha256(),
            );
            Ok(())
        },
    )?;
    assert_eq!(validator_calls, 1);
    assert_eq!(
        accepted.request_identity_sha256(),
        request_b.identity_sha256()
    );
    assert_ne!(accepted.token_identity_sha256(), [0; 32]);
    let lease_b = session_b.record_consumer_completion()?;
    while !lease_b.completion_is_ready()? {
        std::thread::yield_now();
    }
    drop(lease_b);

    let mut session_c = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    let view_c =
        PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let request_c = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&view_c, 0.000_1, 1, 0)?;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = session_c.bind_evaluation_view_with_resident_adaptive_base_checked_v1(
            view_c,
            request_c,
            |_current_token| -> Result<(), ResidentFeatureStoreCudaErrorV3> {
                panic!("validator panic fixture")
            },
        );
    }));
    assert!(unwind.is_err());
    let upload_after_unwind = session_c.upload_scenarios(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ROWS as u32,
        ..ScenarioDescriptor::default()
    }]);
    assert!(matches!(
        upload_after_unwind,
        Err(ResidentFeatureStoreCudaErrorV3::Population(
            CudaPopulationError::Native {
                status: STATUS_STRICT_RESIDENT_POISONED,
                ..
            }
        ))
    ));
    let lease_c = session_c.record_consumer_completion()?;
    while !lease_c.completion_is_ready()? {
        std::thread::yield_now();
    }
    drop(lease_c);
    Ok(())
}

#[cfg(feature = "cuda-device-fixtures")]
#[test]
fn resident_store_v3_terminal_metrics_only_path_is_one_session_and_leak_free()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }

    let admission = acquire_discovery_run_device_admission_v1()?;
    let probes = admission.probe_counters();
    assert_eq!(probes.physical_inventory_probe_count(), 1);
    assert_eq!(probes.cuda_enumeration_count(), 1);
    assert_eq!(probes.primary_context_acquisition_count(), 1);
    assert_eq!(probes.run_stream_creation_count(), 1);

    let run_device = seal_test_full_discovery_run_device_v3(admission, 4 * 1024 * 1024, 1024)?;
    let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
    let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
    let ordinal = run_device.device_identity().ordinal();
    let bindings = fixture_bindings();
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    let materialization = prepare_resident_smc_parent_v3(
        &run_device,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &timestamps,
        bindings.clone(),
    )?;
    let smc_receipt = materialization.receipt();
    assert_eq!(smc_receipt.row_count, ROWS);
    assert_eq!(smc_receipt.feature_column_count, bindings.len());
    assert_eq!(smc_receipt.parent_smc_slot_count, SMC_SLOTS);
    assert_eq!(smc_receipt.producer_ready_event_count, 1);
    assert_eq!(smc_receipt.compact_control_plane_d2h_bytes, 100);
    let working_set = exact_working_set(
        &run_device,
        &bindings,
        smc_receipt.retained_feature_device_bytes,
    )?;

    let (mut assembler, pending) =
        begin_resident_smc_store_v3(run_device, bindings.clone(), &working_set, materialization)?;
    pending.append_to(&mut assembler)?;
    wait_for_batch_retirement(&mut assembler)?;
    let owner = assembler.seal()?;
    let hashes = loop {
        match owner.compact_hashes_if_ready() {
            Ok(hashes) => break hashes,
            Err(ResidentFeatureStoreCudaErrorV3::NotReady) => std::thread::yield_now(),
            Err(error) => return Err(error.into()),
        }
    };
    let layout = owner.layout_evidence(&hashes);
    assert_eq!(layout.rows, ROWS);
    assert_eq!(layout.columns, bindings.len());
    assert_eq!(layout.producer_batch_count, 1);
    assert_eq!(layout.full_feature_major_staging_bytes, 0);

    let resident_import = owner.import_on_consumer_stream(context, stream, ordinal)?;
    let mut session = resident_import.consume_into_population_session_v3()?;
    session.bind_evaluation_view_v1(PopulationEvaluationViewV1::full(
        ROWS,
        PopulationTimestampModeV1::Canonical,
        None,
    )?)?;
    session.bind_evaluation_view_v1(PopulationEvaluationViewV1::contiguous_range(
        ROWS,
        8,
        56,
        PopulationTimestampModeV1::Canonical,
        None,
    )?)?;
    let ordered: Arc<[u64]> = (0_u64..ROWS as u64).step_by(2).collect::<Vec<_>>().into();
    let ordered_rows = ordered.len();
    session.bind_evaluation_view_v1(PopulationEvaluationViewV1::ordered_indices(
        ROWS,
        Arc::clone(&ordered),
        PopulationTimestampModeV1::Canonical,
        None,
    )?)?;
    let adaptive_view =
        PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let adaptive_request =
        ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&adaptive_view, 0.000_1, 1, 0)?;
    session.bind_evaluation_view_with_resident_adaptive_base_v1(adaptive_view, adaptive_request)?;
    let adaptive_base = session.copy_resident_adaptive_base_fixture_v1()?;
    assert_eq!(adaptive_base.len(), ROWS);
    assert!(
        adaptive_base
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    );
    let mut adaptive_hasher = Sha256::new();
    for value in &adaptive_base {
        adaptive_hasher.update(value.to_bits().to_le_bytes());
    }
    let adaptive_sha256: [u8; 32] = adaptive_hasher.finalize().into();
    assert_eq!(
        adaptive_sha256,
        [
            0xf4, 0x07, 0xad, 0x99, 0xd1, 0xbd, 0xc2, 0x38, 0x60, 0x2a, 0xb1, 0xb2, 0x80, 0x65,
            0xe6, 0x1e, 0x82, 0x81, 0x83, 0xe5, 0x6e, 0x97, 0x52, 0x87, 0x59, 0xa1, 0xd9, 0x92,
            0x27, 0x4b, 0xa7, 0x8b,
        ],
        "resident adaptive output must match the canonical CPU exact-v3 byte vector",
    );
    let checkpoint_bits =
        [0_usize, 48, 49, 99, 100, 101, 159].map(|index| adaptive_base[index].to_bits());
    assert_eq!(
        checkpoint_bits,
        [
            0x3e11_2e0b_e826_d695,
            0x3e11_2e0b_e826_d695,
            0x4001_049e_d07d_1db7,
            0x4001_0569_5b28_6744,
            0x4001_0fae_b68c_45f7,
            0x4001_055c_7dca_6cbb,
            0x4001_0569_3d50_cb03,
        ],
    );

    let descriptors = [GeneDescriptor {
        candidate_id: CANDIDATE_ID,
        term_offset: 0,
        term_count: 1,
        long_threshold: 1.0e300,
        short_threshold: -1.0e300,
        stop_ticks: 100,
        target_ticks: 200,
        stop_vol_multiplier: 0.0,
        flags: 0,
        reserved: 0,
    }];
    let offsets = [0_i32, 1];
    let indices = [0_i32];
    let weights = [0.0_f64];
    let stop_pips = [10.0_f64];
    let target_pips = [20.0_f64];
    let stop_vol_multipliers = [0.0_f64];
    let smc_flags = [0_i8; SMC_SLOTS];
    let smc_weights = [0.0_f64; SMC_SLOTS];
    session.upload_genes(PopulationGeneView {
        descriptors: &descriptors,
        offsets: &offsets,
        indices: &indices,
        weights: &weights,
        stop_pips: &stop_pips,
        target_pips: &target_pips,
        stop_vol_multipliers: &stop_vol_multipliers,
        smc_flags: &smc_flags,
        smc_weights: &smc_weights,
        gate_threshold: 1.0e300,
        smc_gate_disabled: true,
    })?;
    session.upload_scenarios(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ordered_rows as u32,
        ..ScenarioDescriptor::default()
    }])?;
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        max_hold_bars: 8,
        min_hold_bars: 1,
        max_trades_per_day: 1,
        month_capacity: 12,
        gap_threshold_ms: 600_000,
        initial_equity: 100_000.0,
        pip_value: 0.000_1,
        spread_pips: 1.0,
        commission_per_trade: 7.0,
        pip_value_per_lot: 10.0,
        risk_per_trade_min: 0.005,
        risk_per_trade_max: 0.01,
        high_quality_confidence: 0.75,
        spread_pips_asian: 1.0,
        spread_pips_overlap: 1.0,
        spread_pips_late_ny: 1.0,
        ..NeoPopulationSettings::default()
    };
    let terminal = session
        .enqueue_metrics_only_v1(&settings)?
        .consume_terminal_compact_result_v1()?;
    assert_eq!(terminal.metric_row().candidate_id, CANDIDATE_ID);
    assert_eq!(terminal.metric_row().scenario_id, SCENARIO_ID);
    assert_eq!(
        terminal.metric_row().values,
        [0.0, 0.0, 100_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(terminal.scenario_count(), 1);
    assert_eq!(terminal.terminal_synchronization_count(), 1);
    assert_eq!(terminal.terminal_readback_count(), 1);
    assert_eq!(terminal.terminal_readback_rows(), 1);
    assert_eq!(terminal.terminal_readback_bytes(), 104);
    assert_ne!(terminal.receipt_identity_sha256(), [0; 32]);

    // A completed strict launch leaves scenario arrays and a metrics-only
    // workspace resident. Uploading the next generation's genes must retire
    // those transients before allocating the new unsplittable gene store; the
    // following multi-scenario launch proves the session remains usable after
    // that exact transition.
    session.upload_genes(PopulationGeneView {
        descriptors: &descriptors,
        offsets: &offsets,
        indices: &indices,
        weights: &weights,
        stop_pips: &stop_pips,
        target_pips: &target_pips,
        stop_vol_multipliers: &stop_vol_multipliers,
        smc_flags: &smc_flags,
        smc_weights: &smc_weights,
        gate_threshold: 1.0e300,
        smc_gate_disabled: true,
    })?;

    let multi_scenarios = [
        ScenarioDescriptor {
            base_candidate_id: 0,
            scenario_id: SCENARIO_ID + 1,
            window_offset: 0,
            window_len: ordered_rows as u32,
            ..ScenarioDescriptor::default()
        },
        ScenarioDescriptor {
            base_candidate_id: 0,
            scenario_id: SCENARIO_ID + 2,
            window_offset: 0,
            window_len: ordered_rows as u32,
            ..ScenarioDescriptor::default()
        },
        ScenarioDescriptor {
            base_candidate_id: 0,
            scenario_id: SCENARIO_ID + 3,
            window_offset: 0,
            window_len: ordered_rows as u32,
            ..ScenarioDescriptor::default()
        },
    ];
    session.upload_scenarios(&multi_scenarios)?;
    let host_metrics = session
        .enqueue_metrics_only_v1(&settings)?
        .consume_host_metrics_v1()?;
    assert_eq!(host_metrics.scenario_count(), 3);
    assert_eq!(host_metrics.terminal_synchronization_count(), 1);
    assert_eq!(host_metrics.terminal_readback_count(), 1);
    assert_eq!(host_metrics.terminal_readback_rows(), 3);
    assert_eq!(host_metrics.terminal_readback_bytes(), 3 * 104);
    assert_ne!(host_metrics.receipt_identity_sha256(), [0; 32]);
    assert_eq!(host_metrics.counters().synchronization_events, 1);
    assert_eq!(host_metrics.counters().full_readback_bytes, 3 * 104);
    let metric_rows = host_metrics.into_metric_rows();
    assert_eq!(metric_rows.len(), multi_scenarios.len());
    for (row, scenario) in metric_rows.iter().zip(multi_scenarios) {
        assert_eq!(row.candidate_id, CANDIDATE_ID);
        assert_eq!(row.scenario_id, scenario.scenario_id);
        assert_eq!(
            row.values,
            [0.0, 0.0, 100_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    let counters = session.read_residency_counters_v1()?;
    assert_eq!(counters.parent_upload_count(), 0);
    assert_eq!(counters.parent_upload_bytes(), 0);
    assert_eq!(counters.view_binding_count(), 4);
    assert_eq!(counters.full_binding_count(), 2);
    assert_eq!(counters.range_binding_count(), 1);
    assert_eq!(counters.ordered_binding_count(), 1);
    assert_eq!(
        counters.ordered_index_upload_bytes(),
        (ordered_rows * std::mem::size_of::<u64>()) as u64
    );
    assert_eq!(counters.adaptive_upload_bytes(), 0);
    assert_eq!(counters.stream_creation_count(), 0);
    assert_eq!(counters.explicit_synchronization_count(), 2);
    assert_eq!(counters.metric_rows_readback_count(), 1);
    assert_eq!(counters.metric_rows_readback_rows(), 3);
    assert_eq!(counters.metric_rows_readback_bytes(), 3 * 104);
    assert_eq!(counters.diagnostic_readback_count(), 1);
    assert_eq!(counters.diagnostic_readback_rows(), ROWS as u64);
    assert_eq!(
        counters.diagnostic_readback_bytes(),
        (ROWS * std::mem::size_of::<f64>()) as u64
    );
    assert_eq!(counters.accepted_trade_total_readback_count(), 0);

    let lease = session.record_consumer_completion()?;
    while !lease.completion_is_ready()? {
        std::thread::yield_now();
    }
    assert_eq!(lease.rows(), ROWS);
    assert_eq!(lease.columns(), bindings.len());
    drop(lease);
    drop(owner);
    Ok(())
}

#[cfg(feature = "cuda-device-fixtures")]
#[test]
fn resident_store_v3_moves_into_search_v2_and_enqueues_on_real_cuda()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }

    let readiness = resident_search_v2_production_readiness();
    assert!(readiness.device_owned_search_control());
    assert!(readiness.native_bridge_production_sealed());
    assert!(!readiness.terminal_cleanup_lease());
    assert!(!readiness.exact_generation_semantics());
    assert!(!readiness.production_ready());

    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    let mut session = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    session.bind_evaluation_view_v1(PopulationEvaluationViewV1::full(
        ROWS,
        PopulationTimestampModeV1::Canonical,
        None,
    )?)?;

    let plan = fixture_search_plan();

    let mut search = session.consume_into_resident_search_run_v2(plan, [1.0; SMC_SLOTS], true)?;
    search.upload_resident_scenarios_v2(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ROWS as u32,
        ..ScenarioDescriptor::default()
    }])?;
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        max_hold_bars: 8,
        min_hold_bars: 1,
        max_trades_per_day: 1,
        month_capacity: 12,
        gap_threshold_ms: 600_000,
        initial_equity: 100_000.0,
        pip_value: 0.000_1,
        spread_pips: 1.0,
        commission_per_trade: 7.0,
        pip_value_per_lot: 10.0,
        risk_per_trade_min: 0.005,
        risk_per_trade_max: 0.01,
        high_quality_confidence: 0.75,
        spread_pips_asian: 1.0,
        spread_pips_overlap: 1.0,
        spread_pips_late_ny: 1.0,
        ..NeoPopulationSettings::default()
    };
    let host_metrics = search
        .enqueue_resident_gene_metrics_fixture_v2(&settings)?
        .consume_host_metrics_v1()?;
    assert_eq!(host_metrics.scenario_count(), 1);
    assert_eq!(host_metrics.terminal_synchronization_count(), 1);
    assert_eq!(host_metrics.terminal_readback_count(), 1);
    assert_eq!(host_metrics.terminal_readback_rows(), 1);
    assert_eq!(host_metrics.terminal_readback_bytes(), 104);
    assert_ne!(host_metrics.receipt_identity_sha256(), [0; 32]);
    let counters = host_metrics.counters();
    assert_eq!(counters.gene_upload_bytes, 0);
    assert!(counters.scenario_upload_bytes > 0);
    assert_eq!(counters.synchronization_events, 1);
    assert_eq!(counters.full_readback_bytes, 104);
    let rows = host_metrics.metric_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].candidate_id, 0);
    assert_eq!(rows[0].scenario_id, SCENARIO_ID);
    assert!(rows[0].values.iter().all(|value| value.is_finite()));

    let lease = search.record_consumer_completion()?;
    while !lease.completion_is_ready()? {
        std::thread::yield_now();
    }
    assert_eq!(lease.rows(), ROWS);
    assert_eq!(lease.columns(), RESIDENT_SMC_COLUMN_NAMES_V3.len());
    drop(lease);
    Ok(())
}

#[test]
fn resident_store_v3_search_start_failure_returns_event_owned_recovery_carrier()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    let session = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    let error = match session.consume_into_resident_search_run_v2(
        fixture_search_plan(),
        [0.0; SMC_SLOTS],
        true,
    ) {
        Ok(_) => return Err("zero SMC weights unexpectedly started Search".into()),
        Err(error) => error,
    };
    assert!(matches!(
        &error,
        ResidentFeatureStoreSearchStartErrorV2::Search { .. }
    ));
    let lease = error
        .into_cleanup_lease()
        .ok_or("Search start failure did not return its recovery lease")?;
    while !lease.completion_is_ready()? {
        std::thread::yield_now();
    }
    assert_eq!(lease.rows(), ROWS);
    assert_eq!(lease.columns(), RESIDENT_SMC_COLUMN_NAMES_V3.len());
    drop(lease);
    Ok(())
}
