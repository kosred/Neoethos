use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
    CanonicalTimeframe, CanonicalVolumeRef, FeatureCellValidity, FeatureColumnF64, Ohlcv,
    publish_canonical_ohlcv_generation,
};
use neoethos_search::data_selection::{
    CanonicalDataSelectionError, CanonicalSearchArtifactEnvelopeV2, CanonicalSearchArtifactScopeV2,
    CanonicalSearchEvaluatedWindowV1, CanonicalSearchInputReceiptV2, CanonicalSearchRunInputV2,
    CanonicalSearchWindowRoleV1, ExactCanonicalSeries,
};

static NEXT_TEMP_ROOT: AtomicU64 = AtomicU64::new(0);

fn receipt_test_frame(
    values: Vec<f64>,
    validity: Vec<FeatureCellValidity>,
) -> neoethos_data::FeatureFrame {
    let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(values.len());
    let column = FeatureColumnF64::new("receipt_exact_bits", values, validity)
        .expect("valid exact-receipt test column");
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(timestamps, vec![column])
        .expect("valid exact-receipt test frame")
}

fn receipt_test_anchor(features: &neoethos_data::FeatureFrame) -> CanonicalDatasetIdentity {
    features.provenance().bindings()[0]
        .dataset_identity()
        .clone()
}

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        let unique = NEXT_TEMP_ROOT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "neoethos-search-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create isolated canonical store");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!(
                "ERROR neoethos-search test cleanup failed for {}: {error}",
                self.0.display()
            );
        }
    }
}

#[test]
fn v2_receipt_identity_binds_every_feature_and_validity_bit() {
    let base_bits = 1.25_f64.to_bits();
    let base = receipt_test_frame(
        vec![0.5, f64::from_bits(base_bits), -0.0],
        vec![FeatureCellValidity::Valid; 3],
    );
    let changed_value = receipt_test_frame(
        vec![0.5, f64::from_bits(base_bits + 1), -0.0],
        vec![FeatureCellValidity::Valid; 3],
    );
    assert_eq!(base.plan_identity(), changed_value.plan_identity());
    assert_eq!(
        base.provenance_identity(),
        changed_value.provenance_identity()
    );

    let anchor = receipt_test_anchor(&base);
    let base_receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &base)
        .expect("exact base receipt");
    let changed_value_receipt =
        CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &changed_value)
            .expect("exact changed-value receipt");
    assert_ne!(
        base_receipt.feature_content_sha256(),
        changed_value_receipt.feature_content_sha256(),
        "one changed f64 bit must change the feature-content identity"
    );
    assert_ne!(
        base_receipt.identity_sha256().unwrap(),
        changed_value_receipt.identity_sha256().unwrap(),
        "one changed f64 bit must change the canonical search-input identity"
    );

    let warmup = receipt_test_frame(
        vec![f64::NAN, f64::NAN, f64::NAN],
        vec![FeatureCellValidity::Warmup; 3],
    );
    let missing = receipt_test_frame(
        vec![f64::NAN, f64::NAN, f64::NAN],
        vec![FeatureCellValidity::MissingInput; 3],
    );
    assert_eq!(warmup.plan_identity(), missing.plan_identity());
    assert_eq!(warmup.provenance_identity(), missing.provenance_identity());
    let warmup_receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &warmup)
        .expect("warmup receipt");
    let missing_receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &missing)
        .expect("missing-input receipt");
    assert_ne!(
        warmup_receipt.feature_content_sha256(),
        missing_receipt.feature_content_sha256(),
        "equal NaN payload bits with different validity codes must not share identity"
    );
}

#[test]
fn v2_receipt_identity_binds_exact_timestamps_and_selected_auto_lane() {
    let frame = receipt_test_frame(vec![0.5, 1.25, -0.0], vec![FeatureCellValidity::Valid; 3]);
    let anchor = receipt_test_anchor(&frame);
    let receipt =
        CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame).expect("exact receipt");

    let mut shifted = frame.clone();
    for timestamp in &mut shifted.timestamps {
        *timestamp += 60_000;
    }
    let shifted_receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &shifted)
        .expect("shifted timestamp receipt");
    assert_ne!(
        receipt.feature_content_sha256(),
        shifted_receipt.feature_content_sha256(),
        "timestamps are part of the exact feature-content identity"
    );

    let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(3);
    let first = FeatureColumnF64::new(
        "receipt_order_a",
        vec![0.5, 1.25, -0.0],
        vec![FeatureCellValidity::Valid; 3],
    )
    .unwrap();
    let second = FeatureColumnF64::new(
        "receipt_order_b",
        vec![2.0, 2.5, 3.0],
        vec![FeatureCellValidity::Valid; 3],
    )
    .unwrap();
    let ordered = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        timestamps.clone(),
        vec![first.clone(), second.clone()],
    )
    .unwrap();
    let reversed = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        timestamps,
        vec![second, first],
    )
    .unwrap();
    let ordered_receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &ordered)
        .expect("ordered-name receipt");
    let reversed_receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &reversed)
        .expect("reversed-name receipt");
    assert_ne!(
        ordered_receipt.feature_content_sha256(),
        reversed_receipt.feature_content_sha256(),
        "ordered feature names are part of the exact feature-content identity"
    );

    let bytes = receipt.to_json_bytes().expect("serialize V2 receipt");
    let mut other_lane: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse V2 receipt JSON");
    assert_eq!(
        other_lane["feature_execution"]["compute_policy"].as_str(),
        Some("auto"),
        "the default feature builder must record its actual Auto dispatch authority"
    );
    let selected = other_lane["feature_execution"]["selected_lane"]
        .as_str()
        .expect("V2 receipt owns the selected feature-math lane");
    let replacement = if selected == "cpu_scalar" {
        "cpu_avx2_fma"
    } else {
        "cpu_scalar"
    };
    other_lane["feature_execution"]["selected_lane"] =
        serde_json::Value::String(replacement.to_owned());
    let other_lane = CanonicalSearchInputReceiptV2::from_json_bytes(
        &serde_json::to_vec(&other_lane).expect("serialize alternate-lane receipt"),
    )
    .expect("alternate supported lane is a structurally valid V2 receipt");
    assert_ne!(
        receipt.identity_sha256().unwrap(),
        other_lane.identity_sha256().unwrap(),
        "two selected Auto lanes must never share a canonical search-input identity"
    );
    assert!(
        other_lane.validate_against(&anchor, &frame).is_err(),
        "a structurally valid but non-executed Auto lane must not bind the current feature bits"
    );

    let mut explicit_cpu: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse V2 receipt JSON");
    explicit_cpu["feature_execution"]["compute_policy"] =
        serde_json::Value::String("cpu_only".to_owned());
    let explicit_cpu = CanonicalSearchInputReceiptV2::from_json_bytes(
        &serde_json::to_vec(&explicit_cpu).expect("serialize explicit-CPU receipt"),
    )
    .expect("same CPU lane under an explicit CPU policy is structurally valid");
    assert_ne!(
        receipt.identity_sha256().unwrap(),
        explicit_cpu.identity_sha256().unwrap(),
        "Auto and explicit CPU policies must not share a receipt identity"
    );
    assert!(
        explicit_cpu.validate_against(&anchor, &frame).is_err(),
        "a receipt cannot relabel Auto-built bits as an explicit CPU-policy execution"
    );

    let mut unknown_math: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse V2 receipt JSON");
    unknown_math["feature_execution"]["vector_ta_math_authority"] =
        serde_json::Value::String("neoethos.vector-ta.cpu-unknown.v0".to_owned());
    assert!(
        CanonicalSearchInputReceiptV2::from_json_bytes(
            &serde_json::to_vec(&unknown_math).expect("serialize unknown-math receipt")
        )
        .is_err(),
        "unrecognized CPU math authority must fail closed"
    );
}

#[test]
fn v2_receipt_rejects_the_legacy_v1_wire_shape_without_migration_authority() {
    let frame = receipt_test_frame(vec![0.5, 1.25, -0.0], vec![FeatureCellValidity::Valid; 3]);
    let anchor = receipt_test_anchor(&frame);
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame)
        .expect("exact V2 receipt");
    let mut legacy: serde_json::Value =
        serde_json::from_slice(&receipt.to_json_bytes().unwrap()).expect("parse V2 receipt");
    legacy["schema_version"] = serde_json::Value::from(1_u64);
    legacy
        .as_object_mut()
        .expect("receipt JSON object")
        .remove("feature_content_sha256");
    legacy
        .as_object_mut()
        .expect("receipt JSON object")
        .remove("feature_execution");

    assert!(
        CanonicalSearchInputReceiptV2::from_json_bytes(
            &serde_json::to_vec(&legacy).expect("serialize legacy receipt")
        )
        .is_err(),
        "V1 receipts require an explicit offline migration; production must fail closed"
    );
}

fn external_identity(
    namespace: &str,
    symbol: &str,
    timeframe: CanonicalTimeframe,
) -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::external(
        namespace,
        symbol,
        timeframe,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid external identity")
}

fn publish(root: &Path, identity: &CanonicalDatasetIdentity, close_offset: f64) {
    let period_ms = identity
        .timeframe()
        .fixed_duration_ms()
        .expect("test uses fixed-duration timeframe");
    let rows = 512_i64;
    let start_ms = 1_704_067_200_000_i64;
    let timestamp = (0..rows)
        .map(|row| start_ms + row * period_ms)
        .collect::<Vec<_>>();
    let close = (0..rows)
        .map(|row| close_offset + row as f64 * 0.000_01)
        .collect::<Vec<_>>();
    let open = close.iter().map(|value| value - 0.000_02).collect();
    let high = close.iter().map(|value| value + 0.000_05).collect();
    let low = close.iter().map(|value| value - 0.000_05).collect();
    let ohlcv = Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: None,
    };
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.search-selection-test.v1",
        identity.canonical_bytes(),
    )
    .expect("valid test provenance");
    publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity,
        expected_generation: None,
        provenance: &provenance,
        ohlcv: &ohlcv,
        volume: CanonicalVolumeRef::Absent,
        rows_per_chunk: 128,
    })
    .expect("publish canonical test generation");
}

#[test]
fn exact_anchor_owns_base_features_and_every_provenance_binding() {
    let root = TempRoot::new("exact-anchor");
    let selected = external_identity("broker-a", "EURUSD", CanonicalTimeframe::M1);
    let other = external_identity("broker-b", "EURUSD", CanonicalTimeframe::M1);
    publish(root.path(), &selected, 1.10);
    publish(root.path(), &other, 9.90);

    let series = ExactCanonicalSeries::open(root.path(), selected.clone())
        .expect("select exact canonical series");
    let input = series
        .load_search_input(&[])
        .expect("build features only from the selected direct generation");

    assert_eq!(input.anchor_identity(), &selected);
    assert_eq!(input.base_frame().artifact().identity(), &selected);
    assert!(input.base_frame().ohlcv().close[0] < 2.0);
    assert_eq!(
        input.base_frame().ohlcv().timestamp.as_deref(),
        Some(input.features().timestamps.as_slice()),
        "the search must not truncate unequal inputs to their shorter length"
    );
    assert!(
        input
            .features()
            .provenance()
            .bindings()
            .iter()
            .all(|binding| binding.dataset_identity() == &selected),
        "feature provenance mixed another source/account into the selected series"
    );
    let receipt = input
        .receipt()
        .expect("build and validate exact search input receipt");
    assert_eq!(receipt.schema_version(), 2);
    assert_eq!(
        receipt.anchor_dataset_identity(),
        selected.to_path_component()
    );
    assert_eq!(receipt.source_bindings().len(), 1);
    assert_eq!(
        receipt.source_bindings()[0].dataset_identity(),
        selected.to_path_component()
    );
    assert_eq!(
        receipt.source_bindings()[0].generation_id(),
        input.base_frame().artifact().generation_id()
    );
    assert_eq!(receipt.feature_plan_identity().len(), 64);
    assert_eq!(receipt.feature_provenance_identity().len(), 64);
    assert_eq!(receipt.source_bindings()[0].segments().len(), 1);
    assert_eq!(receipt.source_bindings()[0].segments()[0].row_start(), 0);
    assert_eq!(
        receipt.source_bindings()[0].segments()[0].row_end(),
        input.features().n_samples() as u64
    );
    receipt
        .validate_against(input.anchor_identity(), input.features())
        .expect("receipt must exactly describe the verified feature input");
    assert_eq!(
        receipt.identity_sha256().expect("receipt identity").len(),
        64
    );
}

#[test]
fn search_receipt_is_strict_versioned_and_rejects_generation_substitution() {
    let root = TempRoot::new("receipt-substitution");
    let selected = external_identity("broker-a", "EURUSD", CanonicalTimeframe::M1);
    publish(root.path(), &selected, 1.10);
    let input = ExactCanonicalSeries::open(root.path(), selected.clone())
        .expect("select exact canonical series")
        .load_search_input(&[])
        .expect("load exact canonical search input");
    let receipt = input.receipt().expect("verified receipt");
    let receipt_id = receipt.identity_sha256().expect("receipt SHA-256");

    let bytes = receipt.to_json_bytes().expect("serialize strict receipt");
    let reopened = CanonicalSearchInputReceiptV2::from_json_bytes(&bytes)
        .expect("reopen strict versioned receipt");
    assert_eq!(reopened, receipt);
    assert_eq!(
        reopened.identity_sha256().expect("reopened identity"),
        receipt_id
    );

    let mut substituted: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse receipt fixture");
    substituted["source_bindings"][0]["generation_id"] =
        serde_json::Value::String("foreign-generation".to_owned());
    let substituted_bytes = serde_json::to_vec(&substituted).expect("serialize substitution");
    let substituted = CanonicalSearchInputReceiptV2::from_json_bytes(&substituted_bytes)
        .expect("the substituted receipt is structurally valid but not input-valid");
    assert_ne!(
        substituted.identity_sha256().expect("substituted identity"),
        receipt_id
    );
    let error = substituted
        .validate_against(input.anchor_identity(), input.features())
        .expect_err("a different generation must never validate against the loaded values");
    assert!(error.to_string().contains("generation"), "{error}");

    let mut unknown_field: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse receipt fixture");
    unknown_field["legacy_symbol"] = serde_json::Value::String("EURUSD".to_owned());
    let unknown_bytes = serde_json::to_vec(&unknown_field).expect("serialize unknown field");
    assert!(
        CanonicalSearchInputReceiptV2::from_json_bytes(&unknown_bytes).is_err(),
        "unknown compatibility fields must fail closed"
    );
}

#[test]
fn runnable_search_input_owns_the_receipt_and_rejects_foreign_ohlcv_values() {
    let root = TempRoot::new("runnable-receipt");
    let selected = external_identity("broker-a", "EURUSD", CanonicalTimeframe::M1);
    let foreign = external_identity("broker-b", "EURUSD", CanonicalTimeframe::M1);
    publish(root.path(), &selected, 1.10);
    publish(root.path(), &foreign, 9.90);
    let input = ExactCanonicalSeries::open(root.path(), selected)
        .expect("select exact canonical series")
        .load_search_input(&[])
        .expect("load exact canonical search input");
    let receipt = input.receipt().expect("verified receipt");
    let runnable =
        CanonicalSearchRunInputV2::new(receipt.clone(), input.features(), input.base_frame())
            .expect("bind receipt to values");
    assert_eq!(runnable.receipt(), &receipt);
    assert_eq!(runnable.features().n_samples(), runnable.ohlcv().len());

    let foreign_input = ExactCanonicalSeries::open(root.path(), foreign)
        .expect("select foreign canonical series")
        .load_search_input(&[])
        .expect("load foreign canonical search input");
    let error =
        CanonicalSearchRunInputV2::new(receipt, input.features(), foreign_input.base_frame())
            .expect_err("same-sized timestamps from a foreign immutable generation must fail");
    assert!(error.to_string().contains("identity"), "{error}");
}

#[test]
fn runnable_multitimeframe_input_binds_each_direct_generation_without_derivation() {
    let root = TempRoot::new("runnable-direct-mtf");
    let base = external_identity("broker-a", "EURUSD", CanonicalTimeframe::M1);
    let higher = external_identity("broker-a", "EURUSD", CanonicalTimeframe::H1);
    publish(root.path(), &base, 1.10);
    publish(root.path(), &higher, 1.20);

    let input = ExactCanonicalSeries::open(root.path(), base)
        .expect("select exact canonical series")
        .load_search_input(&[CanonicalTimeframe::H1])
        .expect("load both direct generations without resampling");
    let receipt = input.receipt().expect("verified multi-timeframe receipt");
    let runnable = CanonicalSearchRunInputV2::new(receipt, input.features(), input.base_frame())
        .expect("bind multi-timeframe receipt to exact base values");

    assert_eq!(runnable.receipt().source_bindings().len(), 2);
    assert!(
        runnable
            .receipt()
            .source_bindings()
            .iter()
            .any(|binding| binding.dataset_identity() == higher.to_path_component()),
        "the direct H1 generation must remain in the runnable receipt"
    );
}

#[test]
fn artifact_scope_binds_exact_receipt_role_and_source_window() {
    let root = TempRoot::new("artifact-scope");
    let selected = external_identity("broker-a", "EURUSD", CanonicalTimeframe::M1);
    publish(root.path(), &selected, 1.10);
    let input = ExactCanonicalSeries::open(root.path(), selected)
        .expect("select exact canonical series")
        .load_search_input(&[])
        .expect("load exact canonical search input");
    let receipt = input.receipt().expect("verified receipt");
    let runnable =
        CanonicalSearchRunInputV2::new(receipt.clone(), input.features(), input.base_frame())
            .expect("bind exact immutable search input");
    let timestamps = runnable.ohlcv().timestamp.as_deref().expect("timestamps");
    let expected_window = CanonicalSearchEvaluatedWindowV1::new(
        CanonicalSearchWindowRoleV1::InSample,
        0,
        400,
        timestamps[0],
        timestamps[399],
    )
    .expect("valid in-sample window");
    let scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::InSample,
        &runnable,
        0..400,
    )
    .expect("bind exact in-sample artifact scope");
    assert_eq!(scope.receipt(), &receipt);
    assert_eq!(scope.evaluated_window(), &expected_window);
    assert_eq!(scope.receipt_sha256(), receipt.identity_sha256().unwrap());
    assert_eq!(scope.identity_sha256().expect("scope identity").len(), 64);
    scope
        .validate_against(&receipt, &expected_window)
        .expect("scope must match the expected receipt and window");

    let bytes = scope.to_json_bytes().expect("serialize scope");
    let reopened = CanonicalSearchArtifactScopeV2::from_json_bytes(&bytes)
        .expect("reopen strict artifact scope");
    assert_eq!(reopened, scope);
    assert_eq!(reopened.schema_version(), 2);

    let mut legacy_scope: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse strict scope");
    legacy_scope["schema_version"] = serde_json::Value::from(1_u64);
    assert!(
        CanonicalSearchArtifactScopeV2::from_json_bytes(
            &serde_json::to_vec(&legacy_scope).expect("serialize legacy scope")
        )
        .is_err(),
        "V1 artifact scopes require explicit offline migration and must fail closed"
    );

    let envelope = CanonicalSearchArtifactEnvelopeV2::new(
        "neoethos.search-test.v1",
        scope.clone(),
        "fnv64:0123456789abcdef",
        vec![1_u64, 2, 3],
    )
    .expect("create receipt-bound artifact envelope");
    let envelope_bytes = envelope.to_json_bytes().expect("serialize envelope");
    let reopened_envelope: CanonicalSearchArtifactEnvelopeV2<Vec<u64>> =
        CanonicalSearchArtifactEnvelopeV2::from_json_bytes(&envelope_bytes)
            .expect("reopen strict receipt-bound envelope");
    assert_eq!(reopened_envelope, envelope);
    assert_eq!(reopened_envelope.schema_version(), 2);
    let mut legacy_envelope: serde_json::Value =
        serde_json::from_slice(&envelope_bytes).expect("parse strict envelope");
    legacy_envelope["schema_version"] = serde_json::Value::from(1_u64);
    assert!(
        CanonicalSearchArtifactEnvelopeV2::<Vec<u64>>::from_json_bytes(
            &serde_json::to_vec(&legacy_envelope).expect("serialize legacy envelope")
        )
        .is_err(),
        "V1 artifact envelopes require explicit offline migration and must fail closed"
    );
    reopened_envelope
        .validate_against(
            "neoethos.search-test.v1",
            "fnv64:0123456789abcdef",
            &receipt,
            &expected_window,
        )
        .expect("exact artifact kind/config/receipt/window must validate");
    assert_eq!(
        reopened_envelope.search_config_hash(),
        "fnv64:0123456789abcdef"
    );
    assert!(
        reopened_envelope
            .validate_against(
                "neoethos.search-test.v1",
                "fnv64:fedcba9876543210",
                &receipt,
                &expected_window,
            )
            .is_err(),
        "an artifact from another resolved search configuration must fail closed"
    );
    assert!(
        reopened_envelope
            .validate_against(
                "neoethos.other-test.v1",
                "fnv64:0123456789abcdef",
                &receipt,
                &expected_window,
            )
            .is_err(),
        "one payload kind must never be reinterpreted as another"
    );

    let wrong_role = CanonicalSearchEvaluatedWindowV1::new(
        CanonicalSearchWindowRoleV1::Holdout,
        0,
        400,
        timestamps[0],
        timestamps[399],
    )
    .expect("valid but semantically different window");
    assert!(
        scope.validate_against(&receipt, &wrong_role).is_err(),
        "an in-sample artifact must not validate as holdout evidence"
    );

    let outside = CanonicalSearchEvaluatedWindowV1::new(
        CanonicalSearchWindowRoleV1::InSample,
        0,
        900,
        timestamps[0],
        timestamps[399],
    )
    .expect("structurally valid window");
    assert!(
        CanonicalSearchArtifactScopeV2::new(receipt.clone(), outside).is_err(),
        "rows outside the receipt source segments must fail closed"
    );

    let mut foreign_receipt_json: serde_json::Value =
        serde_json::from_slice(&receipt.to_json_bytes().expect("serialize receipt"))
            .expect("parse receipt");
    foreign_receipt_json["source_bindings"][0]["generation_id"] =
        serde_json::Value::String("foreign-generation".to_owned());
    let foreign_receipt = CanonicalSearchInputReceiptV2::from_json_bytes(
        &serde_json::to_vec(&foreign_receipt_json).expect("serialize foreign receipt"),
    )
    .expect("foreign generation is structurally valid");
    assert!(
        scope.validate_against_receipt(&foreign_receipt).is_err(),
        "a copied artifact must not validate under another generation receipt"
    );

    let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).expect("parse scope");
    unknown["legacy_symbol"] = serde_json::Value::String("EURUSD".to_owned());
    assert!(
        CanonicalSearchArtifactScopeV2::from_json_bytes(
            &serde_json::to_vec(&unknown).expect("serialize unknown field")
        )
        .is_err(),
        "unknown compatibility authority must fail closed"
    );
}

#[test]
fn missing_direct_higher_timeframe_is_typed_and_lists_the_wrong_source_candidate() {
    let root = TempRoot::new("missing-direct-htf");
    let selected = external_identity("broker-a", "EURUSD", CanonicalTimeframe::M1);
    let wrong_source_h1 = external_identity("broker-b", "EURUSD", CanonicalTimeframe::H1);
    publish(root.path(), &selected, 1.10);
    publish(root.path(), &wrong_source_h1, 9.90);

    let series = ExactCanonicalSeries::open(root.path(), selected.clone())
        .expect("select exact canonical series");
    let error = series
        .load_search_input(&[CanonicalTimeframe::H1])
        .expect_err("M1 must never be resampled into a missing direct H1 generation");

    match error {
        CanonicalDataSelectionError::MissingDirectTimeframe {
            anchor_id,
            requested_symbol,
            requested_timeframe,
            candidate_ids,
        } => {
            assert_eq!(anchor_id, selected.to_path_component());
            assert_eq!(requested_symbol, "EURUSD");
            assert_eq!(requested_timeframe, CanonicalTimeframe::H1);
            assert_eq!(candidate_ids, vec![wrong_source_h1.to_path_component()]);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn unavailable_anchor_is_typed_and_lists_current_symbol_candidates() {
    let root = TempRoot::new("missing-anchor");
    let selected = external_identity("broker-a", "EURUSD", CanonicalTimeframe::M1);
    let current_other = external_identity("broker-b", "EURUSD", CanonicalTimeframe::M1);
    publish(root.path(), &current_other, 9.90);

    let error = ExactCanonicalSeries::open(root.path(), selected.clone())
        .expect_err("a display-symbol match cannot replace the exact requested anchor");
    assert_eq!(
        error,
        CanonicalDataSelectionError::AnchorUnavailable {
            anchor_id: selected.to_path_component(),
            candidate_ids: vec![current_other.to_path_component()],
        }
    );
}

#[test]
fn unique_related_symbol_stays_inside_the_selected_external_namespace() {
    let root = TempRoot::new("related-external-source");
    let anchor = external_identity("broker-a", "EURUSD", CanonicalTimeframe::M1);
    let selected_bridge = external_identity("broker-a", "GBPUSD", CanonicalTimeframe::H1);
    let wrong_source_bridge = external_identity("broker-b", "GBPUSD", CanonicalTimeframe::H1);
    publish(root.path(), &anchor, 1.10);
    publish(root.path(), &selected_bridge, 1.25);
    publish(root.path(), &wrong_source_bridge, 8.80);

    let series =
        ExactCanonicalSeries::open(root.path(), anchor).expect("select exact canonical series");
    let resolved = series
        .select_related_direct("GBPUSD", &[CanonicalTimeframe::H1])
        .expect("one direct bridge exists in the selected namespace");

    assert_eq!(resolved, selected_bridge);
    assert_ne!(resolved, wrong_source_bridge);
}

#[test]
fn related_symbol_selection_is_source_scoped_and_ambiguity_is_typed() {
    use neoethos_data::CTraderEnvironment;
    use neoethos_dataset_contracts::CanonicalDatasetScope;

    let root = TempRoot::new("related-ambiguity");
    let anchor = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "Broker-Demo",
        42,
        1001,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("anchor identity");
    let bridge_one = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "Broker-Demo",
        42,
        2001,
        "GBPUSD",
        CanonicalTimeframe::H1,
        BarTimestampConvention::BarOpen,
    )
    .expect("bridge identity one");
    let bridge_two = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "Broker-Demo",
        42,
        2002,
        "GBPUSD",
        CanonicalTimeframe::H1,
        BarTimestampConvention::BarOpen,
    )
    .expect("bridge identity two");
    let wrong_account = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "Broker-Demo",
        99,
        2003,
        "GBPUSD",
        CanonicalTimeframe::H1,
        BarTimestampConvention::BarOpen,
    )
    .expect("wrong-account bridge identity");
    publish(root.path(), &anchor, 1.10);
    publish(root.path(), &bridge_one, 1.25);
    publish(root.path(), &bridge_two, 1.26);
    publish(root.path(), &wrong_account, 8.80);

    let series = ExactCanonicalSeries::open(root.path(), anchor.clone())
        .expect("select exact canonical series");
    let error = series
        .select_related_direct("GBPUSD", &[CanonicalTimeframe::H1])
        .expect_err("two symbols in the same broker account need explicit disambiguation");

    match error {
        CanonicalDataSelectionError::AmbiguousDirectTimeframe {
            requested_symbol,
            requested_timeframe,
            candidate_ids,
            ..
        } => {
            assert_eq!(requested_symbol, "GBPUSD");
            assert_eq!(requested_timeframe, CanonicalTimeframe::H1);
            assert_eq!(
                candidate_ids,
                vec![
                    bridge_one.to_path_component(),
                    bridge_two.to_path_component()
                ]
            );
            assert!(!candidate_ids.contains(&wrong_account.to_path_component()));
        }
        other => panic!("unexpected error: {other}"),
    }

    assert!(matches!(
        anchor.scope(),
        CanonicalDatasetScope::CTrader { account_id: 42, .. }
    ));
}

#[test]
fn production_search_sources_expose_no_symbol_only_raw_or_resampling_loader() {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let retired = [
        "load_symbol_dataset(",
        "load_symbol_dataset_with_timeframes(",
        "load_symbol_timeframe(",
        "load_symbol_timeframe_tail(",
        "discover_timeframes(",
        "ensure_timeframes_with_resample",
        "load_vortex(",
    ];
    let mut violations = Vec::new();
    let mut sources = Vec::new();
    collect_rust_sources(&source_root, &mut sources);
    for path in sources {
        let source = fs::read_to_string(&path).expect("read search source");
        for needle in retired {
            if source.contains(needle) {
                violations.push(format!("{}: {needle}", path.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "production search still exposes retired data paths:\n{}",
        violations.join("\n")
    );
}

fn collect_rust_sources(root: &Path, sources: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("read search source directory") {
        let entry = entry.expect("read search source entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, sources);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            sources.push(path);
        }
    }
}

#[test]
fn holdout_boundary_never_truncates_misaligned_rows_by_position() {
    let discovery = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("discovery.rs"),
    )
    .expect("read discovery source");
    assert!(
        !discovery.contains("ohlcv.close.len().min(features.n_samples())"),
        "the holdout boundary still hides row-count mismatches by taking the shorter input"
    );
    assert!(
        discovery.contains("features.timestamps.as_slice() == base_timestamps"),
        "the holdout boundary must compare exact timestamps before any positional slice"
    );
}

#[test]
fn diagnostic_examples_require_exact_canonical_identity_and_no_retired_loader() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let exact_data_examples = [
        "gpu_discovery_probe.rs",
        "htf_effective_n_probe.rs",
        "htf_prefilter_probe.rs",
        "tail_cliff_probe.rs",
    ];
    let retired_calls = [
        "load_symbol_dataset(",
        "load_symbol_dataset_with_timeframes(",
        "load_symbol_timeframe(",
        "load_symbol_timeframe_tail(",
        "discover_timeframes(",
        "ensure_timeframes_with_resample",
        "load_vortex(",
        "set_store_root(",
    ];
    for file in exact_data_examples {
        let source = fs::read_to_string(examples.join(file)).expect("read diagnostic example");
        for retired_call in retired_calls {
            assert!(
                !source.contains(retired_call),
                "{file} still reaches retired data API {retired_call}"
            );
        }
        assert!(
            source.contains("--identity"),
            "{file} must require an opaque exact dataset identity"
        );
        assert!(
            source.contains("CanonicalDatasetIdentity::from_path_component"),
            "{file} must decode the exact canonical identity instead of reconstructing it from display fields"
        );
        assert!(
            source.contains("ExactCanonicalSeries::open"),
            "{file} must verify the exact selected series and list current candidates when it is unavailable"
        );
    }
}
