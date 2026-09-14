//! Transport regressions use the real producer for the large payload and the
//! existing small Discovery fixture for exhaustive malformed-wire cases.
use super::*;
use crate::promotion_candidate_training_v1_tests::{
    discovery_fixture_portfolio, exact_receipt, exact_series, handoff,
};
use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::core::normalization::normalize_search_feature_column_f64;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
    CanonicalVolumeRef, FeatureBuildOptions, Ohlcv, SearchNormalizationFittedStateV1,
    load_dataset_for_identity_with_timeframes, prepare_multitimeframe_features_raw_with_options,
    prepare_multitimeframe_features_with_fitted_normalization, publish_canonical_ohlcv_generation,
};
use neoethos_search::CanonicalTrendbarResearchCostAssumptionsV2;
use neoethos_search::data_selection::CanonicalSearchArtifactScopeV2;
use neoethos_search::live_portfolio::LivePortfolioArtifact;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// Independent copy of the pre-V2 derive order, deliberately not the new wire
// DTO. Both optional-holdout cases must reproduce these exact original bytes.
#[derive(Serialize)]
struct LegacyLockedOracle<'a> {
    schema: &'a str,
    version: u16,
    canonical_json: &'a str,
    identity_sha256: &'a str,
}

#[derive(Serialize)]
struct LegacyHandoffOracle<'a> {
    schema: &'a str,
    version: u16,
    canonical_series: &'a CanonicalDatasetSeriesReceiptV1,
    #[serde(serialize_with = "serialize_timeframe_v1")]
    base_timeframe: CanonicalTimeframe,
    search_input_receipt: &'a CanonicalSearchInputReceiptV2,
    screening_contract: &'a CanonicalTrendbarResearchExecutionContractV3,
    locked_portfolio: LegacyLockedOracle<'a>,
    oos_cutoff_ms: i64,
    purge_bars: usize,
    broker_authority: &'a PromotionCandidateBrokerAuthorityIdentityV1,
    training_config: &'a PromotionCandidateTrainingConfigIdentityV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    discovery_holdout_scope: Option<&'a CanonicalSearchArtifactScopeV2>,
}

fn legacy_oracle<'a>(
    value: &'a PromotionCandidateTrainingHandoffV1,
    full_portfolio_json: &'a str,
) -> LegacyHandoffOracle<'a> {
    LegacyHandoffOracle {
        schema: HANDOFF_SCHEMA_V1,
        version: 1,
        canonical_series: &value.canonical_series,
        base_timeframe: value.base_timeframe,
        search_input_receipt: &value.search_input_receipt,
        screening_contract: &value.screening_contract,
        locked_portfolio: LegacyLockedOracle {
            schema: LOCKED_PORTFOLIO_SCHEMA_V1,
            version: 1,
            canonical_json: full_portfolio_json,
            identity_sha256: value.locked_portfolio.identity_sha256(),
        },
        oos_cutoff_ms: value.oos_cutoff_ms,
        purge_bars: value.purge_bars,
        broker_authority: &value.broker_authority,
        training_config: &value.training_config,
        discovery_holdout_scope: value.discovery_holdout_scope.as_ref(),
    }
}

fn screening(
    receipt: &CanonicalSearchInputReceiptV2,
) -> CanonicalTrendbarResearchExecutionContractV3 {
    CanonicalTrendbarResearchExecutionContractV3::new(
        receipt.clone(),
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "neoethos.test.compact-handoff.v1",
            assumption_source_sha256: &"5".repeat(64),
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.0,
            slippage_pips_per_fill_assumption: 0.1,
            commission_account_per_lot_per_fill_assumption: 3.5,
            swap_long_pips_per_day: -0.2,
            swap_short_pips_per_day: -0.1,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .unwrap()
}

fn compact_fixture() -> (LivePortfolioArtifact, PromotionCandidateTrainingHandoffV1) {
    let receipt = exact_receipt();
    let (portfolio, settings) =
        discovery_fixture_portfolio(receipt.clone(), vec!["close_minus_open".to_owned()], false);
    let selected = PromotionCandidateTrainingHandoffV1::from_discovery_portfolio(
        exact_series(&receipt),
        screening(&receipt),
        &portfolio,
        &settings,
    )
    .unwrap();
    (portfolio, selected)
}

#[test]
fn shared_handoff_decodes_once_per_validation_but_revalidates_each_public_boundary() {
    let before_constructor = SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get());
    let (mut portfolio, selected) = compact_fixture();
    // The constructor validates its fresh body directly; only the final full
    // handoff validation needs to attach the encoded body once.
    assert_eq!(
        SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get()),
        before_constructor + 1
    );
    let before = SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get());
    let locked = PromotionCandidateLockedPortfolioV1::from_live_portfolio(&portfolio).unwrap();
    assert_eq!(locked, selected.locked_portfolio);
    assert_eq!(
        SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get()),
        before
    );
    selected.validate().unwrap();
    assert_eq!(
        SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get()),
        before + 1
    );
    selected.to_json_bytes().unwrap();
    assert_eq!(
        SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get()),
        before + 2
    );
    selected.identity_sha256().unwrap();
    assert_eq!(
        SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get()),
        before + 3
    );
    let (bytes, identity) = selected.canonical_bytes_and_identity_sha256().unwrap();
    assert_eq!(
        SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get()),
        before + 4
    );
    assert_eq!(bytes, serde_json::to_vec(&selected).unwrap());
    assert_eq!(
        identity,
        domain_sha256_v1(HANDOFF_IDENTITY_DOMAIN_V2, &bytes)
    );
    let (validated, live_identity) = selected
        .validated_live_portfolio_and_identity_sha256()
        .unwrap();
    assert_eq!(
        SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get()),
        before + 5
    );
    assert_eq!(live_identity, identity);
    assert_eq!(
        serde_json::to_vec(&validated).unwrap(),
        serde_json::to_vec(&portfolio).unwrap()
    );
    portfolio.final_holdout_scope = portfolio.sizing_evidence[0].forward_test.scope().clone();
    assert!(PromotionCandidateLockedPortfolioV1::from_live_portfolio(&portfolio).is_err());
}

#[test]
fn legacy_discovery_handoff_still_validates_semantics_after_exact_json_hash() {
    let (mut portfolio, mut selected) = compact_fixture();
    selected.schema = HANDOFF_SCHEMA_V1.to_owned();
    selected.version = SCHEMA_VERSION_V1;
    selected.locked_portfolio =
        PromotionCandidateLockedPortfolioV1::from_serializable(&portfolio).unwrap();
    selected.validate().unwrap();

    // Recompute the exact JSON identity so only the semantic check can catch
    // the forbidden reuse of calibration as the reserved final test.
    portfolio.final_holdout_scope = portfolio.sizing_evidence[0].forward_test.scope().clone();
    selected.locked_portfolio =
        PromotionCandidateLockedPortfolioV1::from_serializable(&portfolio).unwrap();
    assert_eq!(
        selected.validate().unwrap_err().code(),
        PromotionCandidateTrainingRefusalCodeV1::OosCutoffLeakage
    );
    for code in [
        selected.to_json_bytes().unwrap_err().code(),
        selected.identity_sha256().unwrap_err().code(),
        selected
            .canonical_bytes_and_identity_sha256()
            .unwrap_err()
            .code(),
        selected
            .validated_live_portfolio_and_identity_sha256()
            .unwrap_err()
            .code(),
    ] {
        assert_eq!(
            code,
            PromotionCandidateTrainingRefusalCodeV1::OosCutoffLeakage
        );
    }
}

#[test]
fn handoffs_without_discovery_scope_still_validate_the_locked_payload() {
    let mut generic = handoff(&["alpha"]);
    assert!(generic.discovery_holdout_scope.is_none());
    // A valid generic handoff is not silently reinterpreted as a live artifact.
    assert!(generic.canonical_bytes_and_identity_sha256().is_ok());
    assert!(
        generic
            .validated_live_portfolio_and_identity_sha256()
            .is_err()
    );
    generic.locked_portfolio.canonical_json.push(' ');
    assert_eq!(
        generic.validate().unwrap_err().code(),
        PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff
    );

    let (_, mut shared) = compact_fixture();
    shared.discovery_holdout_scope = None;
    shared.oos_cutoff_ms = shared
        .search_input_receipt
        .source_bindings()
        .iter()
        .flat_map(|binding| binding.segments())
        .map(|segment| segment.timestamp_end_ms())
        .max()
        .unwrap()
        + 1;
    let before = SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get());
    shared.validate().unwrap();
    assert_eq!(
        SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get()),
        before + 1
    );
    shared
        .validated_live_portfolio_and_identity_sha256()
        .unwrap();
    assert_eq!(
        SHARED_PORTFOLIO_DECODE_COUNT.with(|count| count.get()),
        before + 2
    );
    shared.locked_portfolio.identity_sha256 = "0".repeat(64);
    assert_eq!(
        shared.validate().unwrap_err().code(),
        PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff
    );
    assert!(shared.canonical_bytes_and_identity_sha256().is_err());
    assert!(
        shared
            .validated_live_portfolio_and_identity_sha256()
            .is_err()
    );

    // The locked payload itself fits exactly. Only the complete handoff is
    // oversized, so both counting and retaining boundaries must enforce 8MiB.
    generic.locked_portfolio = PromotionCandidateLockedPortfolioV1::from_serializable(
        &"x".repeat(MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 - 2),
    )
    .unwrap();
    assert_eq!(
        generic.locked_portfolio.canonical_json.len(),
        MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1
    );
    for code in [
        generic.validate().unwrap_err().code(),
        generic.to_json_bytes().unwrap_err().code(),
        generic.identity_sha256().unwrap_err().code(),
        generic
            .canonical_bytes_and_identity_sha256()
            .unwrap_err()
            .code(),
        generic
            .validated_live_portfolio_and_identity_sha256()
            .unwrap_err()
            .code(),
    ] {
        assert_eq!(
            code,
            PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge
        );
    }
}

#[test]
fn legacy_handoff_transport_keeps_original_field_order_bytes_and_identity() {
    let mut legacy_with_holdout = compact_fixture().1;
    let portfolio = legacy_with_holdout
        .locked_portfolio
        .deserialize_live_portfolio()
        .unwrap();
    legacy_with_holdout.schema = HANDOFF_SCHEMA_V1.to_owned();
    legacy_with_holdout.version = 1;
    legacy_with_holdout.locked_portfolio =
        PromotionCandidateLockedPortfolioV1::from_serializable(&portfolio).unwrap();
    for legacy in [handoff(&["alpha"]), legacy_with_holdout] {
        let expected = serde_json::to_vec(&legacy_oracle(
            &legacy,
            &legacy.locked_portfolio.canonical_json,
        ))
        .unwrap();
        let actual = legacy.to_json_bytes().unwrap();
        assert_eq!(actual, expected);
        assert!(!String::from_utf8_lossy(&actual).contains("shared_receipt"));
        assert_eq!(
            legacy.identity_sha256().unwrap(),
            domain_sha256_v1(HANDOFF_IDENTITY_DOMAIN_V1, &expected)
        );
        let (combined_bytes, combined_identity) =
            legacy.canonical_bytes_and_identity_sha256().unwrap();
        assert_eq!(combined_bytes, expected);
        assert_eq!(
            combined_identity,
            domain_sha256_v1(HANDOFF_IDENTITY_DOMAIN_V1, &expected)
        );
        if legacy.discovery_holdout_scope.is_some() {
            let (validated, identity) = legacy
                .validated_live_portfolio_and_identity_sha256()
                .unwrap();
            assert_eq!(identity, combined_identity);
            assert_eq!(
                serde_json::to_vec(&validated).unwrap(),
                legacy.locked_portfolio.canonical_json.as_bytes()
            );
        }
        let restored: PromotionCandidateTrainingHandoffV1 =
            serde_json::from_slice(&actual).unwrap();
        assert_eq!(restored.to_json_bytes().unwrap(), expected);
        assert_eq!(
            restored.identity_sha256().unwrap(),
            legacy.identity_sha256().unwrap()
        );
    }
}

#[test]
fn shared_handoff_rejects_receipt_substitution_mixed_versions_and_nested_receipts() {
    let (portfolio, selected) = compact_fixture();
    let bytes = selected.to_json_bytes().unwrap();
    let json = String::from_utf8(bytes).unwrap();
    let original: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(original["locked_portfolio"].get("shared_receipt").is_none());
    assert!(
        original["screening_contract"]
            .get("input_receipt")
            .is_none()
    );
    assert!(original["discovery_holdout_scope"].get("receipt").is_none());
    let body: serde_json::Value = serde_json::from_str(
        original["locked_portfolio"]["canonical_json"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert!(body["search_scope"].get("receipt").is_none());
    assert!(body["sizing_evidence"][0]["scope"].get("receipt").is_none());
    for case in 0..13 {
        let mut changed = original.clone();
        match case {
            0 => changed["schema"] = HANDOFF_SCHEMA_V1.into(),
            1 => changed["version"] = 1.into(),
            2 => {
                changed["schema"] = HANDOFF_SCHEMA_V1.into();
                changed["version"] = 1.into();
                changed["locked_portfolio"]["schema"] = LOCKED_PORTFOLIO_SCHEMA_V1.into();
                changed["locked_portfolio"]["version"] = 1.into();
            }
            3 => changed["locked_portfolio"]["shared_receipt"] = serde_json::Value::Null,
            4 => {
                changed["screening_contract"] =
                    serde_json::to_value(selected.screening_contract()).unwrap()
            }
            5 => {
                changed["discovery_holdout_scope"] =
                    serde_json::to_value(selected.discovery_holdout_scope.as_ref().unwrap())
                        .unwrap()
            }
            6 => changed["screening_contract"]["input_receipt_sha256"] = "0".repeat(64).into(),
            7 => changed["discovery_holdout_scope"]["receipt_sha256"] = "0".repeat(64).into(),
            8 => changed["discovery_holdout_scope"]["scope_sha256"] = "0".repeat(64).into(),
            9 => changed["search_input_receipt"]["feature_content_sha256"] = "0".repeat(64).into(),
            10 => changed["schema"] = "neoethos.unknown.v3".into(),
            11 => changed["unknown_authority"] = true.into(),
            12 => changed["screening_contract"]["schema_version"] = 2.into(),
            _ => unreachable!(),
        }
        assert!(
            serde_json::from_value::<PromotionCandidateTrainingHandoffV1>(changed).is_err(),
            "malformed shared wire case {case} was accepted"
        );
    }
    let duplicate = json.replacen('{', "{\"version\":2,", 1);
    assert!(serde_json::from_str::<PromotionCandidateTrainingHandoffV1>(&duplicate).is_err());
    let duplicate_ref = json.replacen(
        "\"screening_contract\":{",
        "\"screening_contract\":{\"schema_version\":1,",
        1,
    );
    assert!(serde_json::from_str::<PromotionCandidateTrainingHandoffV1>(&duplicate_ref).is_err());

    // The transport deliberately defers portfolio content validation, but no
    // altered compact body may produce a usable handoff or a sealed identity.
    for field in ["portfolio_identity_sha256", "search_config_hash"] {
        let mut changed = original.clone();
        let mut changed_body = body.clone();
        changed_body[field] = "0".repeat(64).into();
        changed["locked_portfolio"]["canonical_json"] =
            serde_json::to_string(&changed_body).unwrap().into();
        let decoded: PromotionCandidateTrainingHandoffV1 = serde_json::from_value(changed).unwrap();
        assert!(
            decoded.identity_sha256().is_err(),
            "tampered {field} admitted"
        );
        assert!(decoded.to_json_bytes().is_err(), "tampered {field} encoded");
        assert!(
            decoded.canonical_bytes_and_identity_sha256().is_err(),
            "tampered {field} published"
        );
        assert!(
            decoded
                .validated_live_portfolio_and_identity_sha256()
                .is_err(),
            "tampered {field} decoded as validated"
        );
    }
    let restored: PromotionCandidateTrainingHandoffV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.to_json_bytes().unwrap(), json.as_bytes());
    assert_eq!(
        restored.identity_sha256().unwrap(),
        selected.identity_sha256().unwrap()
    );
    assert_eq!(
        serde_json::to_vec(
            &restored
                .locked_portfolio
                .deserialize_live_portfolio()
                .unwrap()
        )
        .unwrap(),
        serde_json::to_vec(&portfolio).unwrap()
    );
}

static NEXT_STORE: AtomicU64 = AtomicU64::new(0);
struct ProducerStore(PathBuf);
impl ProducerStore {
    fn new() -> Self {
        let sequence = NEXT_STORE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "neoethos-compact-handoff-producer-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for ProducerStore {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!(
                "ERROR compact producer fixture cleanup {}: {error}",
                self.0.display()
            );
        }
    }
}

fn production_normalized_fixture() -> (ProducerStore, neoethos_data::FeatureFrame) {
    let root = ProducerStore::new();
    let sample_timestamps = neoethos_data::test_fixtures::ctrader_sample_ohlcv()
        .timestamp
        .unwrap();
    let start = sample_timestamps[0];
    let mut anchor = None;
    // Same independently authored direct series as the Data production-fit
    // regression, with the shared Discovery fixture's 100 real timestamps.
    // M5 is a separate published source; it is not generated by resampling M1.
    for (timeframe, rows, stride) in [
        (CanonicalTimeframe::M1, 100, 1),
        (CanonicalTimeframe::M5, 20, 5),
    ] {
        let base = stride == 1;
        let mut frame = Ohlcv {
            timestamp: Some(Vec::new()),
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Some(Vec::new()),
        };
        for row in 0..rows {
            let t = (row * stride) as f64;
            let px = 1.10 + (t * 0.7).sin() * 0.01 + t * 1e-5;
            frame
                .timestamp
                .as_mut()
                .unwrap()
                .push(start + (row * stride) as i64 * 60_000);
            frame.open.push(px);
            frame.high.push(px + if base { 0.0008 } else { 0.0009 });
            frame.low.push(px - if base { 0.0008 } else { 0.0009 });
            frame
                .close
                .push(px + (t * 0.3).cos() * if base { 0.0004 } else { 0.0003 });
            frame.volume.as_mut().unwrap().push(if base {
                100.0 + (row % 17) as f64
            } else {
                500.0 + (row % 19) as f64
            });
        }
        let identity = CanonicalDatasetIdentity::external(
            "compact-handoff-test",
            "EURUSD",
            timeframe,
            BarTimestampConvention::BarOpen,
        )
        .unwrap();
        let producer = ProducerProvenanceEnvelopeV1::new(
            "neoethos.compact-handoff-test.v1",
            format!("independent-direct-{timeframe}").into_bytes(),
        )
        .unwrap();
        publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
            configured_root: &root.0,
            identity: &identity,
            expected_generation: None,
            provenance: &producer,
            ohlcv: &frame,
            volume: CanonicalVolumeRef::Float64(frame.volume.as_deref().unwrap()),
            rows_per_chunk: 128,
        })
        .unwrap();
        if base {
            anchor = Some(identity);
        }
    }
    let dataset =
        load_dataset_for_identity_with_timeframes(&root.0, &anchor.unwrap(), &["M1", "M5"])
            .unwrap();
    let options = FeatureBuildOptions {
        prefix_base_features: true,
        higher_tfs: vec!["M5".into()],
        normalization_training_rows: Some(0..80),
        drop_columns_without_normalization_training_support: true,
        ..Default::default()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let raw = pool
        .install(|| prepare_multitimeframe_features_raw_with_options(&dataset, "M1", &options))
        .unwrap();
    assert_eq!(raw.timestamps, sample_timestamps);
    let mut columns = raw
        .project_columns(&(0..raw.n_features()).collect::<Vec<_>>(), 0..100)
        .unwrap()
        .columns
        .clone();
    columns.retain(|column| {
        column.validity[..80]
            .iter()
            .any(|validity| validity.is_valid())
    });
    let fits = columns
        .iter_mut()
        .map(|column| normalize_search_feature_column_f64(column, 0..80).unwrap())
        .collect();
    let state = SearchNormalizationFittedStateV1::new(
        columns.iter().map(|column| column.name.clone()).collect(),
        fits,
    )
    .unwrap();
    let normalized = pool
        .install(|| {
            prepare_multitimeframe_features_with_fitted_normalization(
                &dataset, "M1", &options, &state,
            )
        })
        .unwrap();
    assert_eq!(normalized.normalization_fitted_state(), Some(&state));
    assert_eq!(
        normalized.n_features(),
        1_924,
        "actual producer vocabulary changed: review the payload census"
    );
    for (index, expected) in columns.iter().enumerate() {
        let actual = normalized.feature_column(index).unwrap();
        assert_eq!(actual.validity, expected.validity);
        assert!(
            actual
                .values
                .iter()
                .zip(&expected.values)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "frozen fit changed producer column {index}"
        );
    }
    (root, normalized)
}

#[derive(Default)]
struct CountingWriter(usize);
impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn real_1924_feature_handoff_fits_unchanged_cap_with_one_receipt_and_original_v5_hash() {
    let (_root, frame) = production_normalized_fixture();
    let anchor = frame
        .provenance()
        .bindings()
        .iter()
        .find(|binding| binding.dataset_identity().timeframe() == CanonicalTimeframe::M1)
        .unwrap()
        .dataset_identity();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(anchor, &frame).unwrap();
    let (portfolio, settings) =
        discovery_fixture_portfolio(receipt.clone(), frame.names.clone(), true);
    let selected = PromotionCandidateTrainingHandoffV1::from_discovery_portfolio(
        exact_series(&receipt),
        screening(&receipt),
        &portfolio,
        &settings,
    )
    .unwrap();
    let compact = selected.to_json_bytes().unwrap();
    let compact_text = String::from_utf8_lossy(&compact);
    assert_eq!(
        compact_text.matches("feature_plan_canonical_bytes").count(),
        1
    );
    assert!(!compact_text.contains("shared_receipt"));
    assert!(compact.len() < MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1);
    let original_v5 = serde_json::to_string(&portfolio).unwrap();
    let old_identity =
        domain_sha256_v1(LOCKED_PORTFOLIO_IDENTITY_DOMAIN_V1, original_v5.as_bytes());
    assert_eq!(selected.locked_portfolio.identity_sha256(), old_identity);
    assert_eq!(
        canonical_locked_portfolio_identity_sha256_v1(&portfolio).unwrap(),
        old_identity
    );
    let legacy = legacy_oracle(&selected, &original_v5);
    let mut old_size = CountingWriter::default();
    serde_json::to_writer(&mut old_size, &legacy).unwrap();
    assert!(old_size.0 > MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1);
    assert_eq!(
        codec::check_bounded(&legacy).unwrap_err().code(),
        PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge
    );
    assert!(compact.len() * 3 < old_size.0);
    let restored: PromotionCandidateTrainingHandoffV1 = serde_json::from_slice(&compact).unwrap();
    assert_eq!(restored.to_json_bytes().unwrap(), compact);
    assert_eq!(
        restored.identity_sha256().unwrap(),
        selected.identity_sha256().unwrap()
    );
    assert_eq!(restored.search_input_receipt(), &receipt);
    assert_eq!(
        serde_json::to_string(
            &restored
                .locked_portfolio
                .deserialize_live_portfolio()
                .unwrap()
        )
        .unwrap(),
        original_v5
    );
    eprintln!(
        "compact_handoff_actual_producer columns={} receipt_bytes={} legacy_handoff_bytes={} shared_handoff_bytes={} cap_bytes={}",
        frame.n_features(),
        receipt.to_json_bytes().unwrap().len(),
        old_size.0,
        compact.len(),
        MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1
    );
}

#[test]
fn compressed_handoff_keeps_received_bytes_and_rejects_mixed_or_unbound_evidence() {
    let (portfolio, mut selected) = compact_fixture();
    let old_bytes = selected.to_json_bytes().unwrap();
    let old_identity = selected.identity_sha256().unwrap();
    assert_eq!(selected.version, SCHEMA_VERSION_V2);
    selected.compressed_search_input_receipt =
        Some(codec::CompressedReceipt::new(&selected.search_input_receipt).unwrap());
    selected.schema = HANDOFF_SCHEMA_V3.to_owned();
    selected.version = SCHEMA_VERSION_V3;
    let bytes = selected.to_json_bytes().unwrap();
    let identity = selected.identity_sha256().unwrap();
    assert_eq!(
        identity,
        domain_sha256_v1(HANDOFF_IDENTITY_DOMAIN_V3, &bytes)
    );
    assert_ne!(identity, old_identity);
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(wire.get("search_input_receipt").is_none());
    let restored: PromotionCandidateTrainingHandoffV1 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);
    assert_eq!(restored.identity_sha256().unwrap(), identity);
    assert_eq!(
        restored.search_input_receipt(),
        selected.search_input_receipt()
    );
    assert_eq!(
        serde_json::to_vec(
            &restored
                .locked_portfolio
                .deserialize_live_portfolio()
                .unwrap()
        )
        .unwrap(),
        serde_json::to_vec(&portfolio).unwrap()
    );

    // A valid alternate compression level is retained, not silently replaced
    // by this build's preferred compressor bytes on reopen.
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::none());
    encoder
        .write_all(&selected.search_input_receipt.to_json_bytes().unwrap())
        .unwrap();
    let alternate_bytes = encoder.finish().unwrap();
    let mut alternate = wire.clone();
    alternate["compressed_search_input_receipt"]["bytes"] =
        serde_json::to_value(&alternate_bytes).unwrap();
    let alternate: PromotionCandidateTrainingHandoffV1 = serde_json::from_value(alternate).unwrap();
    let alternate_wire = alternate.to_json_bytes().unwrap();
    let alternate_value: serde_json::Value = serde_json::from_slice(&alternate_wire).unwrap();
    assert_eq!(
        alternate_value["compressed_search_input_receipt"]["bytes"],
        serde_json::to_value(alternate_bytes).unwrap()
    );
    let reopened: PromotionCandidateTrainingHandoffV1 =
        serde_json::from_slice(&alternate_wire).unwrap();
    assert_eq!(reopened.to_json_bytes().unwrap(), alternate_wire);
    assert_eq!(
        reopened.identity_sha256().unwrap(),
        alternate.identity_sha256().unwrap()
    );

    for case in 0..6 {
        let mut changed = wire.clone();
        match case {
            0 => {
                changed["search_input_receipt"] =
                    serde_json::to_value(&selected.search_input_receipt).unwrap()
            }
            1 => changed["search_input_receipt"] = serde_json::Value::Null,
            2 => {
                changed
                    .as_object_mut()
                    .unwrap()
                    .remove("compressed_search_input_receipt");
            }
            3 => {
                changed["schema"] = HANDOFF_SCHEMA_V2.into();
                changed["version"] = 2.into();
            }
            4 => changed["compressed_search_input_receipt"]["json_sha256"] = "0".repeat(64).into(),
            5 => changed["screening_contract"]["input_receipt_sha256"] = "0".repeat(64).into(),
            _ => unreachable!(),
        }
        assert!(
            serde_json::from_value::<PromotionCandidateTrainingHandoffV1>(changed).is_err(),
            "case {case}"
        );
    }
    selected.oos_cutoff_ms += 1;
    assert_eq!(
        selected.to_json_bytes().unwrap_err().code(),
        PromotionCandidateTrainingRefusalCodeV1::OosCutoffLeakage
    );

    // The old transport remains byte-identical; no implicit migration on read.
    let legacy: PromotionCandidateTrainingHandoffV1 = serde_json::from_slice(&old_bytes).unwrap();
    assert_eq!(legacy.to_json_bytes().unwrap(), old_bytes);
    assert_eq!(legacy.identity_sha256().unwrap(), old_identity);
}

#[test]
fn real_size_repetitive_plan_uses_v3_without_losing_receipt_or_portfolio() {
    // Runtime-generated, explicitly unverified transport fixture, not another
    // committed giant artifact or a market/evaluator/training proof. A valid
    // typed plan of the observed 7.7 MB scale uses bounded repetitive names.
    // Only 100 x 1024 cells are built; the selected strategy still has one term.
    let timestamps = neoethos_data::test_fixtures::ctrader_sample_ohlcv()
        .timestamp
        .unwrap();
    let columns = (0..1024)
        .map(|index| {
            neoethos_data::FeatureColumnF64::new(
                format!("transport_{index:04}_{}", "recipe_".repeat(540)),
                (0..100).map(|row| (row as f64 * 0.1).sin()).collect(),
                vec![neoethos_data::FeatureCellValidity::Valid; 100],
            )
            .unwrap()
        })
        .collect();
    let frame =
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(timestamps, columns)
            .unwrap();
    assert!(frame.plan().canonical_bytes().len() >= 7_677_128);
    let binding = &frame.provenance().bindings()[0];
    let receipt =
        CanonicalSearchInputReceiptV2::from_feature_frame(binding.dataset_identity(), &frame)
            .unwrap();
    // Match the existing fixture's selected-generation seam without expanding
    // its large numeric array into serde_json::Value. This remains test data.
    let generation = format!("g1-{}.vortex", receipt.source_bindings()[0].vortex_sha256());
    let json = String::from_utf8(receipt.to_json_bytes().unwrap())
        .unwrap()
        .replace(
            "\"generation_id\":\"embedded-fixture-v1\"",
            &format!("\"generation_id\":\"{generation}\""),
        );
    let receipt = CanonicalSearchInputReceiptV2::from_json_bytes(json.as_bytes()).unwrap();
    assert!(json.len() > MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1);
    drop(json);
    let (portfolio, settings) =
        discovery_fixture_portfolio(receipt.clone(), vec![frame.names[0].clone()], false);
    let selected = PromotionCandidateTrainingHandoffV1::from_discovery_portfolio(
        exact_series(&receipt),
        screening(&receipt),
        &portfolio,
        &settings,
    )
    .unwrap();
    assert_eq!(selected.version, SCHEMA_VERSION_V3);
    let (bytes, identity) = selected.canonical_bytes_and_identity_sha256().unwrap();
    assert!(bytes.len() < MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1);
    let restored: PromotionCandidateTrainingHandoffV1 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(restored.search_input_receipt(), &receipt);
    let (restored_portfolio, restored_identity) = restored
        .validated_live_portfolio_and_identity_sha256()
        .unwrap();
    assert_eq!(restored_identity, identity);
    assert_eq!(
        canonical_locked_portfolio_identity_sha256_v1(&restored_portfolio).unwrap(),
        canonical_locked_portfolio_identity_sha256_v1(&portfolio).unwrap()
    );
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);
    eprintln!(
        "compressed_handoff_repetitive_plan plan_bytes={} wire_bytes={} cap_bytes={}",
        frame.plan().canonical_bytes().len(),
        bytes.len(),
        MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1
    );
}
