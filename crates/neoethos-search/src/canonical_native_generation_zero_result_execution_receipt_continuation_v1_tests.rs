#[allow(unused_imports)]
use super::*;
use crate::canonical_native_discovery_request_v1::{
    MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1, MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1,
    MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1,
};
use crate::canonical_trendbar_research::{
    CanonicalTrendbarResearchCostAssumptionsV2, CanonicalTrendbarResearchExecutionContractV3,
};
use crate::data_selection::{
    CANONICAL_VECTOR_TA_CUDA_MATH_AUTHORITY_V1, CanonicalGpuResidentSearchInputReceiptV3,
    CanonicalSearchInputReceiptV2,
};
use sha2::{Digest, Sha256};

pub(super) fn financial_contract_v1()
-> crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3 {
    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &features).unwrap();
    CanonicalTrendbarResearchExecutionContractV3::new(
        receipt,
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "neoethos.test.gen0-result.v1",
            assumption_source_sha256: &"c".repeat(64),
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
    .unwrap()
}

pub(super) fn native_receipt_value_v1(
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
) -> serde_json::Value {
    let cpu = serde_json::to_value(contract.input_receipt()).unwrap();
    let mut source_bindings = cpu["source_bindings"].clone();
    for (index, binding) in source_bindings
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        binding["source_node_id"] = serde_json::json!(format!("native-node-{index}"));
    }
    let anchor = contract.input_receipt().anchor_dataset_identity();
    let anchor_binding = source_bindings
        .as_array()
        .unwrap()
        .iter()
        .find(|binding| binding["dataset_identity"] == anchor)
        .unwrap();
    let row_count: u64 = anchor_binding["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|segment| {
            segment["row_end"].as_u64().unwrap() - segment["row_start"].as_u64().unwrap()
        })
        .sum();
    serde_json::json!({
        "schema_version": 3,
        "anchor_dataset_identity": anchor,
        "feature_plan_identity": "d".repeat(64),
        "feature_provenance_identity": "e".repeat(64),
        "content_merkle_algorithm": "neoethos.canonical-feature-content.merkle.v3",
        "feature_content_merkle_sha256": "f".repeat(64),
        "normalization_fit_sha256": "1".repeat(64),
        "row_count": row_count,
        "column_count": 5,
        "feature_execution": {
            "schema_version": 1,
            "compute_policy": "gpu_only",
            "vector_ta_math_authority": CANONICAL_VECTOR_TA_CUDA_MATH_AUTHORITY_V1,
            "selected_lane": "gpu_cuda_f64_strict"
        },
        "source_bindings": source_bindings
    })
}

pub(super) fn native_receipt_from_value_v1(
    value: &serde_json::Value,
) -> CanonicalGpuResidentSearchInputReceiptV3 {
    CanonicalGpuResidentSearchInputReceiptV3::from_json_bytes(&serde_json::to_vec(value).unwrap())
        .unwrap()
}

pub(super) fn unchecked_native_receipt_from_value_v1(
    value: serde_json::Value,
) -> CanonicalGpuResidentSearchInputReceiptV3 {
    serde_json::from_value(value).unwrap()
}

#[test]
fn contract_inner_forgery_and_file_domain_identity_swap_are_rejected() {
    let contract = financial_contract_v1();
    let domain_identity = contract.identity_sha256().unwrap();
    let file_bytes = serde_json::to_vec(&contract).unwrap();
    let file_sha = format!("{:x}", Sha256::digest(&file_bytes));
    assert_ne!(domain_identity, file_sha);
    validate_contract_evidence_v1(
        &contract,
        &domain_identity,
        &file_sha,
        &domain_identity,
        &file_sha,
    )
    .unwrap();
    assert!(
        validate_contract_evidence_v1(
            &contract,
            &domain_identity,
            &file_sha,
            &file_sha,
            &domain_identity,
        )
        .is_err()
    );

    let mut forged = serde_json::to_value(&contract).unwrap();
    forged["input_receipt_sha256"] = serde_json::json!("9".repeat(64));
    let forged: CanonicalTrendbarResearchExecutionContractV3 =
        serde_json::from_value(forged).unwrap();
    assert!(forged.validate().is_err());
    assert!(
        forged
            .validate_against_receipt(forged.input_receipt())
            .is_err()
    );
    let forged_file_sha = format!("{:x}", Sha256::digest(serde_json::to_vec(&forged).unwrap()));
    assert!(
        validate_contract_evidence_v1(
            &forged,
            &domain_identity,
            &forged_file_sha,
            &domain_identity,
            &forged_file_sha,
        )
        .is_err()
    );
}

#[test]
fn native_source_node_names_may_differ_but_immutable_source_facts_may_not() {
    let contract = financial_contract_v1();
    let projection =
        crate::resident_population_auto_sizing_receipt_v2::
            canonical_pinned_source_projection_from_search_receipt_v1(
                contract.input_receipt(),
            )
            .unwrap();
    let mut native_value = native_receipt_value_v1(&contract);
    let native = native_receipt_from_value_v1(&native_value);
    validate_native_source_projection_v1(&native, &projection)
        .expect("graph-local source node names are excluded");
    native_value["source_bindings"][0]["source_node_id"] =
        serde_json::json!("another-graph-local-node");
    validate_native_source_projection_v1(&native_receipt_from_value_v1(&native_value), &projection)
        .expect("source_node_id is the only ignored projection fact");

    let mut row_count_drift = native_value.clone();
    row_count_drift["row_count"] =
        serde_json::json!(native_value["row_count"].as_u64().unwrap() + 1);
    assert!(
        validate_native_source_projection_v1(
            &unchecked_native_receipt_from_value_v1(row_count_drift),
            &projection,
        )
        .is_err()
    );

    for (pointer, replacement) in [
        (
            "/anchor_dataset_identity",
            serde_json::json!("external--other--m1--bar_open"),
        ),
        (
            "/source_bindings/0/dataset_identity",
            serde_json::json!("external--other--m1--bar_open"),
        ),
        (
            "/source_bindings/0/manifest_schema_id",
            serde_json::json!("drift-schema"),
        ),
        (
            "/source_bindings/0/generation_id",
            serde_json::json!("drift-generation"),
        ),
        (
            "/source_bindings/0/manifest_sha256",
            serde_json::json!("2".repeat(64)),
        ),
        (
            "/source_bindings/0/vortex_sha256",
            serde_json::json!("3".repeat(64)),
        ),
        (
            "/source_bindings/0/bar_timestamp_convention",
            serde_json::json!("bar_close"),
        ),
    ] {
        let mut drift = native_value.clone();
        *drift.pointer_mut(pointer).unwrap() = replacement;
        let drift = unchecked_native_receipt_from_value_v1(drift);
        assert!(validate_native_source_projection_v1(&drift, &projection).is_err());
    }

    for pointer in [
        "/source_bindings/0/segments/0/row_start",
        "/source_bindings/0/segments/0/row_end",
        "/source_bindings/0/segments/0/timestamp_start_ms",
        "/source_bindings/0/segments/0/timestamp_end_ms",
    ] {
        let mut drift = native_value.clone();
        let current = drift.pointer(pointer).unwrap().as_i64().unwrap();
        *drift.pointer_mut(pointer).unwrap() = serde_json::json!(current + 1);
        assert!(
            validate_native_source_projection_v1(
                &unchecked_native_receipt_from_value_v1(drift),
                &projection,
            )
            .is_err()
        );
    }
    for mutate in [
        |value: &mut serde_json::Value| {
            value["source_bindings"].as_array_mut().unwrap().pop();
        },
        |value: &mut serde_json::Value| {
            let duplicate = value["source_bindings"][0].clone();
            value["source_bindings"]
                .as_array_mut()
                .unwrap()
                .push(duplicate);
            value["source_bindings"].as_array_mut().unwrap().reverse();
        },
        |value: &mut serde_json::Value| {
            value["source_bindings"][0]["segments"]
                .as_array_mut()
                .unwrap()
                .pop();
        },
        |value: &mut serde_json::Value| {
            let duplicate = value["source_bindings"][0]["segments"][0].clone();
            value["source_bindings"][0]["segments"]
                .as_array_mut()
                .unwrap()
                .push(duplicate);
            value["source_bindings"][0]["segments"]
                .as_array_mut()
                .unwrap()
                .reverse();
        },
    ] {
        let mut drift = native_value.clone();
        mutate(&mut drift);
        assert!(
            validate_native_source_projection_v1(
                &unchecked_native_receipt_from_value_v1(drift),
                &projection,
            )
            .is_err()
        );
    }
}

#[test]
fn native_source_binding_and_segment_order_are_bound_without_count_drift() {
    let contract = financial_contract_v1();
    let base_projection = crate::resident_population_auto_sizing_receipt_v2::
        canonical_pinned_source_projection_from_search_receipt_v1(contract.input_receipt())
        .unwrap();
    assert_eq!(base_projection.bindings().len(), 1);

    let anchor = base_projection.anchor_dataset_identity().clone();
    let second_identity = neoethos_data::CanonicalDatasetIdentity::external(
        "embedded-ctrader-fixture-unverified",
        "EURUSD",
        neoethos_data::CanonicalTimeframe::M5,
        neoethos_data::BarTimestampConvention::BarOpen,
    )
    .unwrap();
    assert_ne!(anchor, second_identity);
    let second_segments = vec![
        neoethos_data::CanonicalPinnedSourceSegmentFactsV1::checked_new(0, 1, 10, 10).unwrap(),
        neoethos_data::CanonicalPinnedSourceSegmentFactsV1::checked_new(1, 2, 20, 20).unwrap(),
    ];
    let second_binding = neoethos_data::CanonicalPinnedSourceBindingFactsV1::checked_new(
        second_identity.clone(),
        "neoethos.test.manifest.v1",
        [0x12; 32],
        "generation-two",
        [0x34; 32],
        second_identity.bar_timestamp_convention(),
        second_segments,
    )
    .unwrap();
    let projection =
        neoethos_data::CanonicalPinnedSourceProjectionV1::checked_from_binding_facts_v1(
            anchor,
            base_projection.parent_row_count(),
            vec![base_projection.bindings()[0].clone(), second_binding],
        )
        .unwrap();

    let mut native_value = native_receipt_value_v1(&contract);
    native_value["source_bindings"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({
            "source_node_id": "native-node-two",
            "dataset_identity": second_identity.to_path_component(),
            "manifest_schema_id": "neoethos.test.manifest.v1",
            "manifest_sha256": "12".repeat(32),
            "generation_id": "generation-two",
            "vortex_sha256": "34".repeat(32),
            "bar_timestamp_convention": "bar_open",
            "segments": [
                {
                    "row_start": 0,
                    "row_end": 1,
                    "timestamp_start_ms": 10,
                    "timestamp_end_ms": 10
                },
                {
                    "row_start": 1,
                    "row_end": 2,
                    "timestamp_start_ms": 20,
                    "timestamp_end_ms": 20
                }
            ]
        }));
    validate_native_source_projection_v1(&native_receipt_from_value_v1(&native_value), &projection)
        .expect("two distinct canonical bindings and segments are valid");

    let mut binding_order_drift = native_value.clone();
    binding_order_drift["source_bindings"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(
        validate_native_source_projection_v1(
            &unchecked_native_receipt_from_value_v1(binding_order_drift),
            &projection,
        )
        .is_err()
    );

    let mut segment_order_drift = native_value;
    segment_order_drift["source_bindings"][1]["segments"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(
        validate_native_source_projection_v1(
            &unchecked_native_receipt_from_value_v1(segment_order_drift),
            &projection,
        )
        .is_err()
    );
}

#[test]
fn native_v3_result_caps_reject_oversized_strings_and_source_shapes() {
    let contract = financial_contract_v1();
    let valid = native_receipt_value_v1(&contract);
    validate_result_native_receipt_v3(&native_receipt_from_value_v1(&valid)).unwrap();

    for schema_version in [0_u64, 2, 4] {
        let mut wrong = valid.clone();
        wrong["schema_version"] = serde_json::json!(schema_version);
        assert!(
            validate_result_native_receipt_v3(&unchecked_native_receipt_from_value_v1(wrong))
                .is_err()
        );
    }
    for mutate in [
        |value: &mut serde_json::Value| value["source_bindings"] = serde_json::json!([]),
        |value: &mut serde_json::Value| {
            let binding = value["source_bindings"][0].clone();
            value["source_bindings"] = serde_json::Value::Array(vec![binding; 15]);
        },
        |value: &mut serde_json::Value| {
            value["source_bindings"][0]["segments"] = serde_json::json!([]);
        },
    ] {
        let mut wrong = valid.clone();
        mutate(&mut wrong);
        assert!(
            validate_result_native_receipt_v3(&unchecked_native_receipt_from_value_v1(wrong))
                .is_err()
        );
    }
    for (sources, segments) in [
        (1, 1),
        (
            MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1,
            MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1,
        ),
    ] {
        validate_native_v3_source_shape_counts_v1(sources, segments)
            .expect("count-only V3 shape boundary is admitted without materializing segments");
    }
    for (sources, segments) in [
        (0, 0),
        (1, 0),
        (
            MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1 + 1,
            MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1 + 1,
        ),
        (2, 1),
        (1, MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1 + 1),
        (1, usize::MAX),
    ] {
        assert!(validate_native_v3_source_shape_counts_v1(sources, segments).is_err());
    }

    for pointer in [
        "/anchor_dataset_identity",
        "/source_bindings/0/source_node_id",
        "/source_bindings/0/dataset_identity",
        "/source_bindings/0/manifest_schema_id",
        "/source_bindings/0/generation_id",
        "/source_bindings/0/bar_timestamp_convention",
    ] {
        let mut oversized = valid.clone();
        *oversized.pointer_mut(pointer).unwrap() =
            serde_json::json!("x".repeat(MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1 + 1));
        let oversized = unchecked_native_receipt_from_value_v1(oversized);
        assert!(validate_result_native_receipt_v3(&oversized).is_err());
    }
    for pointer in [
        "/feature_plan_identity",
        "/feature_provenance_identity",
        "/feature_content_merkle_sha256",
        "/normalization_fit_sha256",
        "/source_bindings/0/manifest_sha256",
        "/source_bindings/0/vortex_sha256",
    ] {
        let mut malformed = valid.clone();
        *malformed.pointer_mut(pointer).unwrap() = serde_json::json!("A".repeat(64));
        assert!(
            validate_result_native_receipt_v3(&unchecked_native_receipt_from_value_v1(malformed))
                .is_err()
        );
    }
    for (pointer, replacement) in [
        ("/content_merkle_algorithm", serde_json::json!("wrong")),
        (
            "/feature_execution/compute_policy",
            serde_json::json!("cpu_only"),
        ),
        (
            "/feature_execution/vector_ta_math_authority",
            serde_json::json!("wrong"),
        ),
        (
            "/feature_execution/selected_lane",
            serde_json::json!("cpu_scalar"),
        ),
    ] {
        let mut malformed = valid.clone();
        *malformed.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            validate_result_native_receipt_v3(&unchecked_native_receipt_from_value_v1(malformed))
                .is_err()
        );
    }
    assert_eq!(
        checked_native_v3_receipt_json_upper_bound_bytes_v1(14, 1_000_000).unwrap(),
        175_923_287
    );
}
