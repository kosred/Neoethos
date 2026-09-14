use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64};
use neoethos_data::core::normalization::{
    SEARCH_NORMALIZATION_POLICY_VERSION, normalize_feature_column_f64,
    normalize_search_feature_column_f64,
};
use neoethos_data::core::smc::compute_smc_feature_columns_f64;
use neoethos_data::test_fixtures::{ctrader_sample_ohlcv, ctrader_test_feature_frame_from_columns};
use neoethos_data::{FeatureFrame, Ohlcv};
use neoethos_search::genetic::build_smc_arrays;
use neoethos_search::genetic::smc_indicators::SmcSignalTuple;

const GATES: [&str; 11] = [
    "smc_ob",
    "smc_fvg",
    "smc_liq_sweep",
    "smc_trend_bias",
    "smc_pd_array",
    "smc_inducement",
    "smc_bos",
    "smc_mss",
    "smc_eqh",
    "smc_eql",
    "smc_displacement",
];

fn column(name: &str, values: &[f64]) -> FeatureColumnF64 {
    FeatureColumnF64::new(
        name,
        values.to_vec(),
        vec![FeatureCellValidity::Valid; values.len()],
    )
    .unwrap()
}

fn arrays(tuple: SmcSignalTuple) -> [Vec<i8>; 11] {
    [
        tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7, tuple.8, tuple.9,
        tuple.10,
    ]
}

fn frame_and_bars(columns: Vec<FeatureColumnF64>) -> (FeatureFrame, Ohlcv) {
    let n = columns[0].len();
    let timestamp: Vec<_> = (0..n)
        .map(|i| 1_700_000_100_000 + i as i64 * 60_000)
        .collect();
    let bars = Ohlcv {
        timestamp: Some(timestamp.clone()),
        open: vec![1.0; n],
        high: vec![1.2; n],
        low: vec![0.8; n],
        close: vec![1.0; n],
        volume: Some(vec![1.0; n]),
    };
    (
        ctrader_test_feature_frame_from_columns(timestamp, columns).unwrap(),
        bars,
    )
}

#[test]
fn common_event_and_constant_no_event_are_states_not_centered_measurements() {
    assert_eq!(SEARCH_NORMALIZATION_POLICY_VERSION, 3);
    for name in GATES {
        for prefix in ["", "H1_", "MN1_"] {
            let name = format!("{prefix}{name}");
            for values in [&[1.0, 1.0, 1.0, 0.0][..], &[0.0, 0.0, 0.0, 1.0][..]] {
                let mut feature = column(&name, values);
                let fit = normalize_search_feature_column_f64(&mut feature, 0..3).unwrap();
                assert_eq!(fit.median, 0.0);
                assert_eq!(fit.scale, 1.0);
                assert!(!fit.degenerate);
                assert_eq!(feature.values, values, "{name}");
                assert!(feature.validity.iter().all(|v| v.is_valid()));
            }
        }
    }
}

#[test]
fn normalization_retains_both_event_directions_and_typed_invalidity() {
    let mut feature = FeatureColumnF64::new(
        "smc_inducement",
        vec![1.0, 1.0, f64::NAN, 1.0, -1.0, 0.0],
        vec![
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Gap,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
        ],
    )
    .unwrap();
    let fit = normalize_search_feature_column_f64(&mut feature, 0..4).unwrap();
    assert_eq!(fit.valid_training_cells, 3);
    assert_eq!(feature.validity[2], FeatureCellValidity::Gap);
    assert_eq!(feature.values[2].to_bits(), f64::NAN.to_bits());
    assert_eq!(&feature.values[3..], &[1.0, -1.0, 0.0]);
}

#[test]
fn atr_scaled_trend_remains_zero_anchored_and_bounded() {
    let mut feature = column("smc_trend_bias", &[2.0, 3.0, 4.0, -0.1, 0.0, 100.0, -100.0]);
    normalize_search_feature_column_f64(&mut feature, 0..3).unwrap();
    assert_eq!(feature.values, [2.0, 3.0, 4.0, -0.1, 0.0, 10.0, -10.0]);
}

#[test]
fn continuous_smc_and_decoy_names_keep_the_exact_robust_transform() {
    for name in [
        "smc_fvg_strength",
        "smc_fib_618",
        "smc_asian_range",
        "rsi",
        "unknown_smc_ob",
        "smc_ob_strength",
        "obv",
    ] {
        let mut expected = column(name, &[1.0, 2.0, 3.0, -10.0, 20.0]);
        let mut actual = expected.clone();
        let fit = normalize_feature_column_f64(&mut expected, 0..3).unwrap();
        let actual_fit = normalize_search_feature_column_f64(&mut actual, 0..3).unwrap();
        assert_eq!(fit, actual_fit, "{name}");
        assert_eq!(actual.values, expected.values, "{name}");
        assert_eq!(actual.validity, expected.validity, "{name}");
    }
}

#[test]
fn invalid_state_and_missing_training_support_are_refused_without_mutation() {
    for (name, bad) in [("smc_bos", 0.25), ("H1_smc_eqh", -1.0)] {
        let mut feature = column(name, &[1.0, 0.0, bad]);
        let before = feature.values.clone();
        let error = normalize_search_feature_column_f64(&mut feature, 0..2).unwrap_err();
        assert!(error.to_string().contains("invalid raw state"));
        assert_eq!(feature.values, before);
    }
    let mut missing = FeatureColumnF64::new(
        "smc_ob",
        vec![f64::NAN, 1.0],
        vec![FeatureCellValidity::Warmup, FeatureCellValidity::Valid],
    )
    .unwrap();
    assert!(normalize_search_feature_column_f64(&mut missing, 0..1).is_err());
    assert!(normalize_search_feature_column_f64(&mut missing, 0..3).is_err());
}

#[test]
fn canonical_column_directions_match_the_producer_not_the_alias_spelling() {
    let mut columns: Vec<_> = GATES.iter().map(|name| column(name, &[0.0; 3])).collect();
    columns[4] = column("smc_pd_array", &[1.0, -1.0, 0.0]);
    columns[5] = column("smc_inducement", &[-1.0, 1.0, 0.0]);
    columns[7] = column("smc_mss", &[-1.0, 1.0, 0.0]);
    columns[8] = column("smc_eqh", &[1.0, 0.0, 1.0]);
    columns[9] = column("smc_eql", &[1.0, 1.0, 0.0]);
    let (frame, bars) = frame_and_bars(columns);
    let actual = arrays(build_smc_arrays(&frame, &bars).unwrap());
    assert_eq!(
        actual[4],
        [-1, 1, 0],
        "premium means short, discount means long"
    );
    assert_eq!(
        actual[5],
        [1, 1, 0],
        "either inducement direction is present"
    );
    assert_eq!(actual[7], [-1, 1, 0], "MSS direction is unchanged");
    assert_eq!(
        actual[8],
        [-1, 0, -1],
        "equal highs are the short-side gate"
    );
    assert_eq!(actual[9], [1, 1, 0], "equal lows are the long-side gate");
    assert_eq!(
        actual[2], [0; 3],
        "equal levels do not fabricate a liquidity sweep"
    );
}

#[test]
fn a_different_gate_cannot_replace_an_absent_or_invalid_primary_event() {
    let mut columns: Vec<_> = GATES.iter().map(|name| column(name, &[0.0; 3])).collect();
    columns[0] = FeatureColumnF64::new(
        "smc_ob",
        vec![0.0, f64::NAN, 0.0],
        vec![
            FeatureCellValidity::Valid,
            FeatureCellValidity::Warmup,
            FeatureCellValidity::Valid,
        ],
    )
    .unwrap();
    columns[6] = column("smc_bos", &[1.0, -1.0, 1.0]);
    columns[7] = column("smc_mss", &[-1.0, 1.0, -1.0]);
    columns[10] = column("smc_displacement", &[1.0, -1.0, 1.0]);
    let (frame, bars) = frame_and_bars(columns);
    let actual = arrays(build_smc_arrays(&frame, &bars).unwrap());
    for index in [0, 1, 2, 3, 5] {
        assert_eq!(
            actual[index], [0; 3],
            "{} was fabricated by another signal",
            GATES[index]
        );
    }
}

#[test]
fn actual_smc_producer_to_normalized_frame_to_search_preserves_all_gate_votes() {
    let bars = ctrader_sample_ohlcv();
    let raw = compute_smc_feature_columns_f64(&bars)
        .unwrap()
        .into_iter()
        .filter(|c| GATES.contains(&c.name.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(raw.len(), GATES.len());
    let timestamp = bars.timestamp.clone().unwrap();
    let raw_frame =
        ctrader_test_feature_frame_from_columns(timestamp.clone(), raw.clone()).unwrap();
    let mut normalized = raw;
    for feature in &mut normalized {
        normalize_search_feature_column_f64(feature, 0..80).unwrap();
    }
    let normalized_frame = ctrader_test_feature_frame_from_columns(timestamp, normalized).unwrap();
    assert_eq!(
        arrays(build_smc_arrays(&raw_frame, &bars).unwrap()),
        arrays(build_smc_arrays(&normalized_frame, &bars).unwrap())
    );
}

#[test]
fn malformed_frame_and_ohlc_lengths_are_errors_not_partial_gate_arrays() {
    let columns = GATES.iter().map(|name| column(name, &[0.0; 3])).collect();
    let (frame, mut bars) = frame_and_bars(columns);
    bars.high.pop();
    assert!(
        build_smc_arrays(&frame, &bars)
            .unwrap_err()
            .to_string()
            .contains("length mismatch")
    );
    bars.close.pop();
    assert!(
        build_smc_arrays(&frame, &bars)
            .unwrap_err()
            .to_string()
            .contains("row count mismatch")
    );
}
