use crate::data_selection::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
};
use crate::eval::{BacktestSettings, SmcRow};
use crate::exact_resident_dataset_authority_v1::{
    EXACT_RESIDENT_DATASET_AUTHORITY_SCHEMA_VERSION_V1,
    ExactResidentDatasetAuthorityDeriveRequestV1, ExactResidentDatasetAuthorityErrorCodeV1,
    ExactResidentDatasetParentSealRequestV1, ExactResidentDatasetViewRequestV1,
    SealedExactResidentDatasetParentV1, derive_exact_resident_dataset_authority_v1,
    seal_exact_resident_dataset_parent_v1,
};
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame, Ohlcv};

fn feature_frame(rows: usize, changed_row: Option<usize>) -> FeatureFrame {
    let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(rows);
    let body = (0..rows)
        .map(|row| {
            let value = 1.0 + row as f64 * 0.000_01;
            if changed_row == Some(row) {
                f64::from_bits(value.to_bits() + 1)
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let range = (0..rows)
        .map(|row| 0.000_2 + row as f64 * 0.000_001)
        .collect::<Vec<_>>();
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        timestamps,
        vec![
            FeatureColumnF64::new(
                "resident_body",
                body,
                vec![FeatureCellValidity::Valid; rows],
            )
            .expect("body column"),
            FeatureColumnF64::new(
                "resident_range",
                range,
                vec![FeatureCellValidity::Valid; rows],
            )
            .expect("range column"),
        ],
    )
    .expect("exact resident feature fixture")
}

fn scope(features: &FeatureFrame) -> CanonicalSearchArtifactScopeV2 {
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, features)
        .expect("exact canonical receipt");
    CanonicalSearchArtifactScopeV2::for_entire_receipt(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        receipt,
    )
    .expect("exact canonical scope")
}

fn ohlcv(timestamps: &[i64], changed_close_row: Option<usize>) -> Ohlcv {
    let close = (0..timestamps.len())
        .map(|row| {
            let value = 1.10 + row as f64 * 0.000_01;
            if changed_close_row == Some(row) {
                f64::from_bits(value.to_bits() + 1)
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    Ohlcv {
        timestamp: Some(timestamps.to_vec()),
        open: close.iter().map(|value| value - 0.000_01).collect(),
        high: close.iter().map(|value| value + 0.000_10).collect(),
        low: close.iter().map(|value| value - 0.000_10).collect(),
        close,
        volume: Some(
            (0..timestamps.len())
                .map(|row| 100.0 + row as f64)
                .collect(),
        ),
    }
}

fn priced_settings() -> BacktestSettings {
    BacktestSettings {
        pip_value: 0.000_1,
        spread_pips: 1.2,
        commission_per_trade: 7.0,
        pip_value_per_lot: 10.0,
        ..BacktestSettings::default()
    }
}

fn seal(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    smc: &[SmcRow],
    settings: &BacktestSettings,
    view: ExactResidentDatasetViewRequestV1<'_>,
) -> crate::exact_resident_dataset_authority_v1::ExactResidentDatasetAuthorityV1 {
    let parent = seal_parent(features, ohlcv, smc);
    derive_exact_resident_dataset_authority_v1(ExactResidentDatasetAuthorityDeriveRequestV1 {
        parent: &parent,
        settings,
        view,
    })
    .expect("derive exact resident authority")
}

fn seal_parent(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    smc: &[SmcRow],
) -> SealedExactResidentDatasetParentV1 {
    seal_exact_resident_dataset_parent_v1(ExactResidentDatasetParentSealRequestV1 {
        scope: &scope(features),
        features,
        ohlcv,
        smc_data: smc,
    })
    .expect("seal exact resident parent")
}

#[test]
fn parent_identity_hashes_every_row_not_a_strided_sample() {
    let rows = 600;
    let features = feature_frame(rows, None);
    let bars = ohlcv(&features.timestamps, None);
    let smc = vec![[0_i8; 11]; rows];
    let settings = priced_settings();
    let baseline = seal(
        &features,
        &bars,
        &smc,
        &settings,
        ExactResidentDatasetViewRequestV1::Full,
    );

    let changed_bars = ohlcv(&features.timestamps, Some(413));
    let changed_close = seal(
        &features,
        &changed_bars,
        &smc,
        &settings,
        ExactResidentDatasetViewRequestV1::Full,
    );
    assert_ne!(
        baseline.parent_dataset_identity_sha256(),
        changed_close.parent_dataset_identity_sha256(),
        "one late, formerly-unsampled close bit must invalidate residency"
    );

    let changed_features = feature_frame(rows, Some(417));
    let changed_feature = seal(
        &changed_features,
        &ohlcv(&changed_features.timestamps, None),
        &smc,
        &settings,
        ExactResidentDatasetViewRequestV1::Full,
    );
    assert_ne!(
        baseline.parent_dataset_identity_sha256(),
        changed_feature.parent_dataset_identity_sha256(),
        "one late, formerly-unsampled f64 feature bit must invalidate residency"
    );
}

#[test]
fn parent_identity_binds_validity_smc_open_volume_and_canonical_scope() {
    let rows = 8;
    let features = feature_frame(rows, None);
    let bars = ohlcv(&features.timestamps, None);
    let settings = priced_settings();
    let smc = vec![[0_i8; 11]; rows];
    let baseline = seal(
        &features,
        &bars,
        &smc,
        &settings,
        ExactResidentDatasetViewRequestV1::Full,
    );

    let mut changed_smc = smc.clone();
    changed_smc[7][10] = -1;
    let smc_authority = seal(
        &features,
        &bars,
        &changed_smc,
        &settings,
        ExactResidentDatasetViewRequestV1::Full,
    );
    assert_ne!(
        baseline.parent_dataset_identity_sha256(),
        smc_authority.parent_dataset_identity_sha256()
    );

    let mut changed_open = bars.clone();
    changed_open.open[6] = f64::from_bits(changed_open.open[6].to_bits() + 1);
    let open_authority = seal(
        &features,
        &changed_open,
        &smc,
        &settings,
        ExactResidentDatasetViewRequestV1::Full,
    );
    assert_ne!(
        baseline.parent_dataset_identity_sha256(),
        open_authority.parent_dataset_identity_sha256()
    );

    let mut changed_volume = bars.clone();
    changed_volume.volume.as_mut().unwrap()[5] += 1.0;
    let volume_authority = seal(
        &features,
        &changed_volume,
        &smc,
        &settings,
        ExactResidentDatasetViewRequestV1::Full,
    );
    assert_ne!(
        baseline.parent_dataset_identity_sha256(),
        volume_authority.parent_dataset_identity_sha256()
    );

    let timestamps = features.timestamps.clone();
    let validity_changed = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        timestamps,
        vec![
            FeatureColumnF64::new(
                "resident_body",
                vec![f64::NAN; rows],
                vec![FeatureCellValidity::Warmup; rows],
            )
            .unwrap(),
            FeatureColumnF64::new(
                "resident_range",
                vec![f64::NAN; rows],
                vec![FeatureCellValidity::MissingInput; rows],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let validity_authority = seal(
        &validity_changed,
        &ohlcv(&validity_changed.timestamps, None),
        &smc,
        &settings,
        ExactResidentDatasetViewRequestV1::Full,
    );
    assert_ne!(
        baseline.parent_dataset_identity_sha256(),
        validity_authority.parent_dataset_identity_sha256(),
        "typed validity and its receipt/scope binding are identity-bearing"
    );
}

#[test]
fn evaluation_identity_binds_every_setting_and_the_complete_adaptive_series() {
    let rows = 600;
    let features = feature_frame(rows, None);
    let bars = ohlcv(&features.timestamps, None);
    let smc = vec![[0_i8; 11]; rows];
    let baseline_settings = priced_settings();
    let baseline = seal(
        &features,
        &bars,
        &smc,
        &baseline_settings,
        ExactResidentDatasetViewRequestV1::Full,
    );

    let mut changed_spread = baseline_settings.clone();
    changed_spread.spread_pips = f64::from_bits(changed_spread.spread_pips.to_bits() + 1);
    let spread = seal(
        &features,
        &bars,
        &smc,
        &changed_spread,
        ExactResidentDatasetViewRequestV1::Full,
    );
    assert_eq!(
        baseline.parent_dataset_identity_sha256(),
        spread.parent_dataset_identity_sha256()
    );
    assert_ne!(
        baseline.evaluation_binding_sha256(),
        spread.evaluation_binding_sha256()
    );

    let mut adaptive = baseline_settings.clone();
    adaptive.adaptive_base_pips = Some(vec![12.0_f64; rows].into());
    let adaptive_base = seal(
        &features,
        &bars,
        &smc,
        &adaptive,
        ExactResidentDatasetViewRequestV1::Full,
    );
    let mut changed = vec![12.0_f64; rows];
    changed[419] = f64::from_bits(changed[419].to_bits() + 1);
    adaptive.adaptive_base_pips = Some(changed.into());
    let adaptive_changed = seal(
        &features,
        &bars,
        &smc,
        &adaptive,
        ExactResidentDatasetViewRequestV1::Full,
    );
    assert_ne!(
        adaptive_base.evaluation_binding_sha256(),
        adaptive_changed.evaluation_binding_sha256(),
        "adaptive identity must cover every row rather than 256 samples"
    );
}

#[test]
fn range_and_ordered_index_views_are_exact_reusable_descriptors() {
    let rows = 12;
    let features = feature_frame(rows, None);
    let bars = ohlcv(&features.timestamps, None);
    let smc = vec![[0_i8; 11]; rows];
    let settings = priced_settings();

    let range = seal(
        &features,
        &bars,
        &smc,
        &settings,
        ExactResidentDatasetViewRequestV1::ContiguousRange { start: 2, end: 9 },
    );
    assert_eq!(range.view().row_count(), 7);
    assert_eq!(range.view().contiguous_range(), Some(2..9));
    assert!(range.view().ordered_indices().is_none());

    let ordered = [0_usize, 1, 4, 8, 11];
    let indices = seal(
        &features,
        &bars,
        &smc,
        &settings,
        ExactResidentDatasetViewRequestV1::OrderedIndices(&ordered),
    );
    assert_eq!(indices.view().ordered_indices(), Some(ordered.as_slice()));
    assert_ne!(range.view_identity_sha256(), indices.view_identity_sha256());
    assert_eq!(
        range.parent_dataset_identity_sha256(),
        indices.parent_dataset_identity_sha256(),
        "views reuse one exact parent rather than materializing new host datasets"
    );
}

#[test]
fn invalid_views_and_adaptive_lengths_fail_closed() {
    let rows = 8;
    let features = feature_frame(rows, None);
    let bars = ohlcv(&features.timestamps, None);
    let smc = vec![[0_i8; 11]; rows];
    let settings = priced_settings();
    let parent = seal_parent(&features, &bars, &smc);

    for view in [
        ExactResidentDatasetViewRequestV1::ContiguousRange { start: 3, end: 3 },
        ExactResidentDatasetViewRequestV1::ContiguousRange { start: 2, end: 9 },
        ExactResidentDatasetViewRequestV1::OrderedIndices(&[0, 4, 4]),
        ExactResidentDatasetViewRequestV1::OrderedIndices(&[0, 7, 6]),
    ] {
        let error = derive_exact_resident_dataset_authority_v1(
            ExactResidentDatasetAuthorityDeriveRequestV1 {
                parent: &parent,
                settings: &settings,
                view,
            },
        )
        .expect_err("invalid resident view must fail closed");
        assert_eq!(
            error.code(),
            ExactResidentDatasetAuthorityErrorCodeV1::InvalidView
        );
    }

    let mut adaptive = settings;
    adaptive.adaptive_base_pips = Some(vec![10.0_f64; 7].into());
    let error =
        derive_exact_resident_dataset_authority_v1(ExactResidentDatasetAuthorityDeriveRequestV1 {
            parent: &parent,
            settings: &adaptive,
            view: ExactResidentDatasetViewRequestV1::Full,
        })
        .expect_err("partial adaptive series must fail closed");
    assert_eq!(
        error.code(),
        ExactResidentDatasetAuthorityErrorCodeV1::AdaptiveShapeMismatch
    );
}

#[test]
fn adaptive_series_is_explicitly_view_local_for_range_and_index_authorities() {
    let rows = 8;
    let features = feature_frame(rows, None);
    let bars = ohlcv(&features.timestamps, None);
    let smc = vec![[0_i8; 11]; rows];
    let parent = seal_parent(&features, &bars, &smc);

    let mut parent_indexed = priced_settings();
    parent_indexed.adaptive_base_pips = Some(vec![10.0_f64; rows].into());
    for view in [
        ExactResidentDatasetViewRequestV1::ContiguousRange { start: 2, end: 6 },
        ExactResidentDatasetViewRequestV1::OrderedIndices(&[0, 2, 5, 7]),
    ] {
        let error = derive_exact_resident_dataset_authority_v1(
            ExactResidentDatasetAuthorityDeriveRequestV1 {
                parent: &parent,
                settings: &parent_indexed,
                view,
            },
        )
        .expect_err("a parent-length adaptive series must not be reinterpreted as view-local");
        assert_eq!(
            error.code(),
            ExactResidentDatasetAuthorityErrorCodeV1::AdaptiveShapeMismatch
        );
    }

    let mut view_local = priced_settings();
    view_local.adaptive_base_pips = Some(vec![10.0_f64; 4].into());
    let range = seal(
        &features,
        &bars,
        &smc,
        &view_local,
        ExactResidentDatasetViewRequestV1::ContiguousRange { start: 2, end: 6 },
    );
    let indices = seal(
        &features,
        &bars,
        &smc,
        &view_local,
        ExactResidentDatasetViewRequestV1::OrderedIndices(&[0, 2, 5, 7]),
    );
    assert_ne!(
        range.evaluation_binding_sha256(),
        indices.evaluation_binding_sha256(),
        "the view descriptor remains identity-bearing even for equal view-local adaptive bytes"
    );
}

#[test]
fn authority_is_versioned_opaque_and_has_no_mutable_wire_or_scalar_hash_constructor() {
    let rows = 4;
    let features = feature_frame(rows, None);
    let bars = ohlcv(&features.timestamps, None);
    let smc = vec![[0_i8; 11]; rows];
    let authority = seal(
        &features,
        &bars,
        &smc,
        &priced_settings(),
        ExactResidentDatasetViewRequestV1::Full,
    );
    assert_eq!(
        authority.schema_version(),
        EXACT_RESIDENT_DATASET_AUTHORITY_SCHEMA_VERSION_V1
    );

    let source = include_str!("exact_resident_dataset_authority_v1.rs");
    assert!(!source.contains("Deserialize"));
    assert!(!source.contains("impl Default for ExactResidentDatasetAuthorityV1"));
    assert!(!source.contains("caller_supplied_hash"));
    assert!(!source.contains("sample_hash"));
    assert!(
        !source.contains(".to_dense_samples_major()"),
        "resident authority must not allocate the entire dense feature matrix solely to hash it"
    );
    assert!(source.contains("dense_window(start, end)"));
    assert!(source.contains("FEATURE_HASH_MAX_CELLS_V1"));
    assert!(source.contains("pub(crate) fn seal_exact_resident_dataset_parent_v1"));
    assert!(source.contains("pub(crate) fn derive_exact_resident_dataset_authority_v1"));
}

#[test]
fn view_derivation_accepts_only_the_opaque_parent_and_cannot_rehash_source_arrays() {
    let source = include_str!("exact_resident_dataset_authority_v1.rs");
    let start = source
        .find("pub(crate) fn derive_exact_resident_dataset_authority_v1")
        .expect("view authority derive function");
    let derive = &source[start..];

    assert!(derive.contains("request.parent.parent_dataset_identity_sha256"));
    for forbidden in [
        "FeatureFrame",
        "Ohlcv",
        "smc_data",
        "hash_parent(",
        "dense_window(",
        "seal_exact_resident_dataset_parent_v1(",
    ] {
        assert!(
            !derive.contains(forbidden),
            "view derivation must not reopen or rehash parent source arrays: {forbidden}"
        );
    }
}
