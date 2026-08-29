use super::{
    RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3, ResidentFeatureColumnBindingV3,
    ResidentFeatureStoreCudaErrorV3,
};
use crate::full_discovery_workspace_plan_v1::seal_test_full_discovery_run_device_v3;
use crate::population::{
    PopulationEvaluationViewV1, PopulationGeneView, PopulationTimestampModeV1,
};
use crate::resident_smc_v3::{
    RESIDENT_SMC_COLUMN_NAMES_V3, begin_resident_smc_store_v3, prepare_resident_smc_parent_v3,
};
use crate::{
    GeneDescriptor, NeoPopulationSettings, SMC_SLOTS, ScenarioDescriptor,
    acquire_discovery_run_device_admission_v1,
};
use neoethos_gpu_contracts::ABI_VERSION;
use neoethos_gpu_contracts::resident_feature_store_v3::ResidentWorkingSetRequestV3;
use std::sync::Arc;

const ROWS: usize = 64;
const CANDIDATE_ID: u64 = 7;
const SCENARIO_ID: u64 = 11;

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

    let counters = session.read_residency_counters_v1()?;
    assert_eq!(counters.parent_upload_count(), 0);
    assert_eq!(counters.parent_upload_bytes(), 0);
    assert_eq!(counters.view_binding_count(), 3);
    assert_eq!(counters.full_binding_count(), 1);
    assert_eq!(counters.range_binding_count(), 1);
    assert_eq!(counters.ordered_binding_count(), 1);
    assert_eq!(
        counters.ordered_index_upload_bytes(),
        (ordered_rows * std::mem::size_of::<u64>()) as u64
    );
    assert_eq!(counters.adaptive_upload_bytes(), 0);
    assert_eq!(counters.stream_creation_count(), 0);
    assert_eq!(counters.explicit_synchronization_count(), 0);
    assert_eq!(counters.metric_rows_readback_count(), 0);
    assert_eq!(counters.diagnostic_readback_count(), 0);
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
