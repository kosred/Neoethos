use crate::data_selection::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
};
use crate::engine_identity::PopulationEvalEngine;
use crate::population_engine_run_receipt_v1::{
    POPULATION_ENGINE_RUN_RECEIPT_SCHEMA_VERSION_V1, PopulationEngineRunReceiptErrorCodeV1,
    begin_population_engine_run_v1,
};
use neoethos_data::{FeatureCellValidity, FeatureColumnF64};

fn scope(rows: usize) -> CanonicalSearchArtifactScopeV2 {
    let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(rows);
    let feature = FeatureColumnF64::new(
        "engine_receipt_feature",
        (0..rows).map(|row| row as f64).collect(),
        vec![FeatureCellValidity::Valid; rows],
    )
    .unwrap();
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        timestamps,
        vec![feature],
    )
    .unwrap();
    let anchor = frame.provenance().bindings()[0].dataset_identity().clone();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame).unwrap();
    CanonicalSearchArtifactScopeV2::for_entire_receipt(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        receipt,
    )
    .unwrap()
}

#[test]
fn concurrent_run_scopes_are_isolated_and_count_only_successful_exact_outputs() {
    let scope_a = scope(8);
    let scope_b = scope(9);
    let run_a = begin_population_engine_run_v1(&scope_a).unwrap();
    let run_b = begin_population_engine_run_v1(&scope_b).unwrap();

    let workers = (0..8)
        .map(|_| {
            let run = run_a.clone();
            std::thread::spawn(move || {
                for _ in 0..25 {
                    run.record_successful_population(PopulationEvalEngine::CudaNativeF64, 4, 4)
                        .unwrap();
                }
            })
        })
        .collect::<Vec<_>>();
    run_b
        .record_successful_population(PopulationEvalEngine::Cpu, 3, 3)
        .unwrap();
    for worker in workers {
        worker.join().unwrap();
    }

    let receipt_a = run_a.finish().unwrap();
    let receipt_b = run_b.finish().unwrap();
    assert_eq!(receipt_a.successful_population_count(), 200);
    assert_eq!(receipt_a.engines(), &[PopulationEvalEngine::CudaNativeF64]);
    assert_eq!(receipt_b.successful_population_count(), 1);
    assert_eq!(receipt_b.engines(), &[PopulationEvalEngine::Cpu]);
    assert_ne!(
        receipt_a.canonical_scope_identity_sha256(),
        receipt_b.canonical_scope_identity_sha256()
    );
}

#[test]
fn failed_or_wrong_cardinality_output_never_records_an_engine() {
    let canonical_scope = scope(8);
    let run = begin_population_engine_run_v1(&canonical_scope).unwrap();
    let mismatch = run
        .record_successful_population(PopulationEvalEngine::CudaNativeF64, 4, 3)
        .unwrap_err();
    assert_eq!(
        mismatch.code(),
        PopulationEngineRunReceiptErrorCodeV1::OutputCardinalityMismatch
    );
    let empty = run
        .record_successful_population(PopulationEvalEngine::CubeclF64, 0, 0)
        .unwrap_err();
    assert_eq!(
        empty.code(),
        PopulationEngineRunReceiptErrorCodeV1::EmptyPopulation
    );
    assert_eq!(
        run.finish().unwrap_err().code(),
        PopulationEngineRunReceiptErrorCodeV1::NoSuccessfulPopulation
    );
}

#[test]
fn finish_is_one_way_and_post_finish_recording_fails_closed() {
    let canonical_scope = scope(8);
    let run = begin_population_engine_run_v1(&canonical_scope).unwrap();
    run.record_successful_population(PopulationEvalEngine::Cpu, 2, 2)
        .unwrap();
    let receipt = run.finish().unwrap();
    assert_eq!(
        receipt.schema_version(),
        POPULATION_ENGINE_RUN_RECEIPT_SCHEMA_VERSION_V1
    );
    assert_eq!(
        run.finish().unwrap_err().code(),
        PopulationEngineRunReceiptErrorCodeV1::RunClosed
    );
    assert_eq!(
        run.record_successful_population(PopulationEvalEngine::Cpu, 2, 2)
            .unwrap_err()
            .code(),
        PopulationEngineRunReceiptErrorCodeV1::RunClosed
    );
}

#[test]
fn receipt_engine_order_and_identity_are_deterministic() {
    let canonical_scope = scope(8);
    let run = begin_population_engine_run_v1(&canonical_scope).unwrap();
    run.record_successful_population(PopulationEvalEngine::CubeclF64, 1, 1)
        .unwrap();
    run.record_successful_population(PopulationEvalEngine::Cpu, 1, 1)
        .unwrap();
    run.record_successful_population(PopulationEvalEngine::CubeclF64, 1, 1)
        .unwrap();
    let receipt = run.finish().unwrap();
    assert_eq!(
        receipt.engines(),
        &[PopulationEvalEngine::Cpu, PopulationEvalEngine::CubeclF64]
    );
    assert_eq!(receipt.successful_population_count(), 3);
    assert_eq!(receipt.identity_sha256().len(), 64);
}

#[test]
fn receipt_has_no_global_registry_default_or_mutable_wire_authority() {
    let source = include_str!("population_engine_run_receipt_v1.rs");
    assert!(!source.contains("static OBSERVED"));
    assert!(!source.contains("OnceLock"));
    assert!(!source.contains("Deserialize"));
    assert!(!source.contains("impl Default for PopulationEngineRun"));
    assert!(source.contains("Mutex<PopulationEngineRunStateV1>"));
    assert!(source.contains("pub(crate) fn begin_population_engine_run_v1"));
    assert!(source.contains("pub(crate) fn record_successful_population"));
}
