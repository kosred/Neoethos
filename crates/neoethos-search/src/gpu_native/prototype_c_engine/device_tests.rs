//! Real-adapter tests for the resident Prototype C pipeline.
//!
//! These tests execute. The only tolerated non-execution is a genuinely absent
//! CubeCL adapter, which surfaces as a typed unsupported status and is reported
//! explicitly; any other failure, including a parity mismatch, fails the suite.

use super::create_prototype_c_engine;
use crate::gpu_native::engine::{BacktestEngine, DeviceFilterPolicy, EngineError};
use crate::gpu_native::population_fixture::TinyPopulationFixture;
use crate::gpu_native::prototype_population::{
    PropFirmRequirement, PrototypeBcRequirements, PrototypePopulationWorkload,
};
use crate::gpu_native::prototype_population_oracle::evaluate_population_oracle;

struct DeviceRun {
    summary: crate::gpu_native::engine::HostSurvivorSummary,
    emitted_events: usize,
    transfers: crate::gpu_native::engine::TransferSnapshot,
}

/// Drive one workload through the full engine contract.
///
/// Returns `None` only when this machine has no CubeCL adapter at all.
fn run_on_device(workload: &PrototypePopulationWorkload, session_id: u64) -> Option<DeviceRun> {
    let max_events = workload.genes.population() * workload.dataset.bars();
    let mut engine = match create_prototype_c_engine(None, session_id, max_events) {
        Ok(engine) => engine,
        Err(EngineError::UnsupportedCapability { operation, detail })
            if operation == "prototype_c_gpu_adapter" =>
        {
            eprintln!("Prototype C device test skipped: no CubeCL adapter ({detail})");
            return None;
        }
        Err(error) => panic!("Prototype C session creation failed: {error}"),
    };

    let dataset = engine
        .upload_dataset(&workload.dataset.encode().unwrap())
        .expect("dataset upload");
    let genes = engine
        .upload_genes(&workload.genes.encode().unwrap())
        .expect("gene upload");
    let scenarios = engine
        .upload_scenarios(&workload.scenarios.encode().unwrap())
        .expect("scenario upload");
    let (metrics, evaluate_event) = engine
        .evaluate(dataset, genes, scenarios, None)
        .expect("device evaluation");
    let emitted_events = engine.emitted_events();
    let (selection, filter_event) = engine
        .filter(metrics, DeviceFilterPolicy::All, evaluate_event)
        .expect("device selection");
    let summary = engine
        .readback_compact(selection, filter_event)
        .expect("compact readback");
    let transfers = engine.session().transfer_snapshot();
    Some(DeviceRun {
        summary,
        emitted_events,
        transfers,
    })
}

fn fixed_stop_workload(population: usize, bars: usize) -> PrototypePopulationWorkload {
    TinyPopulationFixture::new(population, bars, 4)
        .population_workload(PrototypeBcRequirements {
            prop_firm_state: PropFirmRequirement::NotRequested,
        })
        .expect("the tiny fixture must produce an eligible B/C workload")
}

fn adaptive_stop_workload(population: usize, bars: usize) -> PrototypePopulationWorkload {
    let fixture = TinyPopulationFixture::new(population, bars, 4);
    let (mut dataset, mut genes, scenarios) = fixture.prototype_a_uploads();
    dataset.settings.adaptive_base_pips = Some(vec![14.0; dataset.bars()]);
    dataset.settings.adaptive_rr = 2.0;
    for (index, multiplier) in genes.stop_vol_multipliers.iter_mut().enumerate() {
        *multiplier = if index % 2 == 0 { 0.0 } else { 1.25 };
    }
    PrototypePopulationWorkload::from_uploads(
        dataset,
        genes,
        scenarios,
        PrototypeBcRequirements {
            prop_firm_state: PropFirmRequirement::NotRequested,
        },
    )
    .expect("the adaptive workload must stay inside the common B/C intersection")
}

fn assert_matches_oracle(workload: &PrototypePopulationWorkload, run: &DeviceRun, label: &str) {
    let expected = evaluate_population_oracle(workload).expect("oracle evaluation");
    let expected_metrics = expected
        .metrics
        .iter()
        .map(|row| row.values)
        .collect::<Vec<_>>();

    assert_eq!(
        run.summary.candidate_ids, workload.genes.candidate_ids,
        "{label}: candidate identity must survive the device round trip"
    );
    assert_eq!(
        run.summary.scenario_ids,
        workload
            .scenarios
            .scenarios
            .iter()
            .map(|scenario| scenario.scenario_id)
            .collect::<Vec<_>>(),
        "{label}: scenario identity must survive the device round trip"
    );
    assert_eq!(
        run.emitted_events,
        expected.events.len(),
        "{label}: device event count must match the canonical emission"
    );

    // A parity assertion over an all-zero workload proves nothing. Require the
    // reference to have actually traded before trusting the comparison.
    assert!(
        expected.counters.accepted_trade_count > 0,
        "{label}: the reference must accept trades for parity to be meaningful"
    );
    assert!(
        expected_metrics
            .iter()
            .any(|row| row[8] > 0.0 && row[0] != 0.0),
        "{label}: the reference must produce non-trivial metrics"
    );

    let report =
        TinyPopulationFixture::compare_final_metrics(&expected_metrics, &run.summary.metrics);
    assert!(
        report.is_match(),
        "{label}: level-10 parity failed: {:?}",
        report.first_divergence
    );
}

#[test]
fn fixed_stop_population_matches_the_canonical_oracle() {
    let workload = fixed_stop_workload(4, 256);
    let Some(run) = run_on_device(&workload, 401) else {
        return;
    };
    assert_matches_oracle(&workload, &run, "fixed stops");

    run.transfers
        .assert_device_resident_chain()
        .expect("one dataset upload, no dense intermediate D2H, no chained re-upload");
    assert_eq!(
        run.transfers.compact_d2h_readbacks, 1,
        "the compact metric readback is the only result boundary"
    );
}

#[test]
fn adaptive_at_entry_population_matches_the_canonical_oracle() {
    let workload = adaptive_stop_workload(4, 256);
    let Some(run) = run_on_device(&workload, 402) else {
        return;
    };
    assert_matches_oracle(&workload, &run, "adaptive-at-entry stops");
}

#[test]
fn gap_exits_and_daily_trade_caps_match_the_canonical_oracle() {
    let fixture = TinyPopulationFixture::new(3, 256, 4);
    let (mut dataset, genes, scenarios) = fixture.prototype_a_uploads();
    dataset.settings.gap_threshold_ms = 120_000;
    dataset.settings.max_trades_per_day = 2;
    // A ten-minute hole inside an otherwise one-minute series forces the gap
    // branch, which has different exit pricing from a stop or target.
    for timestamp in dataset.timestamps.iter_mut().skip(90) {
        *timestamp += 600_000;
    }
    let workload = PrototypePopulationWorkload::from_uploads(
        dataset,
        genes,
        scenarios,
        PrototypeBcRequirements {
            prop_firm_state: PropFirmRequirement::NotRequested,
        },
    )
    .expect("gap and trade-cap settings stay inside the common B/C intersection");

    let Some(run) = run_on_device(&workload, 403) else {
        return;
    };
    assert_matches_oracle(&workload, &run, "gap exits and daily caps");
}

#[test]
fn a_second_evaluation_reuses_the_resident_dataset() {
    let workload = fixed_stop_workload(2, 192);
    let max_events = workload.genes.population() * workload.dataset.bars();
    let mut engine = match create_prototype_c_engine(None, 404, max_events) {
        Ok(engine) => engine,
        Err(EngineError::UnsupportedCapability { operation, .. })
            if operation == "prototype_c_gpu_adapter" =>
        {
            eprintln!("Prototype C residency test skipped: no CubeCL adapter");
            return;
        }
        Err(error) => panic!("Prototype C session creation failed: {error}"),
    };

    let dataset = engine
        .upload_dataset(&workload.dataset.encode().unwrap())
        .expect("dataset upload");
    let genes = engine
        .upload_genes(&workload.genes.encode().unwrap())
        .expect("gene upload");
    let scenarios = engine
        .upload_scenarios(&workload.scenarios.encode().unwrap())
        .expect("scenario upload");

    let mut first = Vec::new();
    for repetition in 0..2 {
        let (metrics, evaluate_event) = engine
            .evaluate(dataset, genes, scenarios, None)
            .expect("device evaluation");
        let (selection, filter_event) = engine
            .filter(metrics, DeviceFilterPolicy::All, evaluate_event)
            .expect("device selection");
        let summary = engine
            .readback_compact(selection, filter_event)
            .expect("compact readback");
        if repetition == 0 {
            first = summary.metrics;
        } else {
            assert_eq!(
                first, summary.metrics,
                "a repeated evaluation on resident buffers must be deterministic"
            );
        }
    }

    let transfers = engine.session().transfer_snapshot();
    assert_eq!(
        transfers.dataset_uploads, 1,
        "a session performs exactly one logical dataset upload"
    );
    assert_eq!(
        transfers.chained_reuploads, 0,
        "chained operations must not re-upload"
    );
    assert_eq!(
        transfers.full_d2h_readbacks, 0,
        "no dense intermediate readback is allowed"
    );
}

#[test]
fn a_second_dataset_upload_is_refused() {
    let workload = fixed_stop_workload(2, 128);
    let mut engine = match create_prototype_c_engine(None, 405, 4096) {
        Ok(engine) => engine,
        Err(EngineError::UnsupportedCapability { operation, .. })
            if operation == "prototype_c_gpu_adapter" =>
        {
            eprintln!("Prototype C reupload test skipped: no CubeCL adapter");
            return;
        }
        Err(error) => panic!("Prototype C session creation failed: {error}"),
    };
    let encoded = workload.dataset.encode().unwrap();
    engine.upload_dataset(&encoded).expect("first upload");
    match engine.upload_dataset(&encoded) {
        Err(EngineError::UnsupportedCapability { operation, .. }) => {
            assert_eq!(operation, "dataset_reupload");
        }
        other => panic!("a second dataset upload must be refused, got {other:?}"),
    }
}

#[test]
fn an_over_capacity_population_is_refused_instead_of_truncated() {
    let workload = fixed_stop_workload(4, 256);
    let mut engine = match create_prototype_c_engine(None, 406, 1) {
        Ok(engine) => engine,
        Err(EngineError::UnsupportedCapability { operation, .. })
            if operation == "prototype_c_gpu_adapter" =>
        {
            eprintln!("Prototype C capacity test skipped: no CubeCL adapter");
            return;
        }
        Err(error) => panic!("Prototype C session creation failed: {error}"),
    };
    let dataset = engine
        .upload_dataset(&workload.dataset.encode().unwrap())
        .expect("dataset upload");
    let genes = engine
        .upload_genes(&workload.genes.encode().unwrap())
        .expect("gene upload");
    let scenarios = engine
        .upload_scenarios(&workload.scenarios.encode().unwrap())
        .expect("scenario upload");

    match engine.evaluate(dataset, genes, scenarios, None) {
        Err(EngineError::UnsupportedCapability { operation, detail }) => {
            assert_eq!(operation, "event_capacity");
            assert!(detail.contains("above the session capacity"), "{detail}");
        }
        other => panic!("an over-capacity population must be refused, got {other:?}"),
    }
}

#[test]
fn an_ineligible_population_is_refused_before_any_device_work() {
    let fixture = TinyPopulationFixture::new(2, 128, 4);
    let (mut dataset, genes, scenarios) = fixture.prototype_a_uploads();
    dataset.settings.trailing_enabled = true;
    dataset.settings.trailing_be_trigger_r = 0.5;

    let mut engine = match create_prototype_c_engine(None, 407, 4096) {
        Ok(engine) => engine,
        Err(EngineError::UnsupportedCapability { operation, .. })
            if operation == "prototype_c_gpu_adapter" =>
        {
            eprintln!("Prototype C eligibility test skipped: no CubeCL adapter");
            return;
        }
        Err(error) => panic!("Prototype C session creation failed: {error}"),
    };
    engine
        .upload_dataset(&dataset.encode().unwrap())
        .expect("dataset upload");
    engine
        .upload_genes(&genes.encode().unwrap())
        .expect("gene upload");
    match engine.upload_scenarios(&scenarios.encode().unwrap()) {
        Err(EngineError::UnsupportedCapability { operation, detail }) => {
            assert_eq!(operation, "population_eligibility");
            assert!(detail.contains("GlobalTrailingOrBreakEven"), "{detail}");
        }
        other => panic!("a trailing population must be refused, got {other:?}"),
    }
}
