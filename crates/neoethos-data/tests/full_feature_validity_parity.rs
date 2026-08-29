use neoethos_data::Ohlcv;
use neoethos_data::core::features::FeatureCellValidity;
use neoethos_data::core::footprint_features::compute_footprint_feature_columns_f64;
use neoethos_data::core::hpc_ta::{IndicatorComputePolicy, compute_classic_ta_feature_columns_f64};
use neoethos_data::core::quant_features::compute_quant_feature_columns_f64;
use neoethos_data::core::regime_detection::compute_regime_feature_columns_f64;
use neoethos_data::core::session_features::compute_session_feature_columns_f64;
use neoethos_data::core::smc::compute_smc_feature_columns_f64;

fn fixture(n: usize, with_volume: bool) -> Ohlcv {
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    let mut timestamps = Vec::with_capacity(n);
    let mut price = 1.1_f64;
    for i in 0..n {
        let next = price + ((i % 5) as f64 - 2.0) * 0.0001;
        open.push(price);
        high.push(price.max(next) + 0.0002 + i as f64 * 1.0e-7);
        low.push(price.min(next) - 0.0002);
        close.push(next);
        volume.push(100.0 + (i % 7) as f64 * 13.0);
        timestamps.push(1_577_836_800_000 + i as i64 * 300_000);
        price = next;
    }
    Ohlcv {
        timestamp: Some(timestamps),
        open,
        high,
        low,
        close,
        volume: with_volume.then_some(volume),
    }
}

fn session_fixture(n: usize, with_volume: bool) -> Ohlcv {
    let mut bars = fixture(n, with_volume);
    bars.timestamp = Some(
        (0..n)
            .map(|row| 1_577_836_800_000 + row as i64 * 3_600_000)
            .collect(),
    );
    bars
}

#[test]
fn classic_vector_ta_missing_volume_invalidates_only_volume_inputs() {
    let columns = compute_classic_ta_feature_columns_f64(
        &fixture(240, false),
        IndicatorComputePolicy::CpuOnly,
        240,
    )
    .expect("validity-aware classic/vector-ta columns");

    for name in ["obv", "mfi", "vwap"] {
        let column = columns
            .iter()
            .find(|column| column.name == name)
            .unwrap_or_else(|| panic!("missing stable-schema volume column {name}"));
        assert!(
            column
                .validity
                .iter()
                .all(|validity| *validity == FeatureCellValidity::MissingInput),
            "{name} treated fabricated zero volume as observations"
        );
        assert!(column.values.iter().all(|value| value.is_nan()));
    }

    let rsi = columns
        .iter()
        .find(|column| column.name == "rsi")
        .expect("price-only RSI");
    assert!(
        rsi.validity
            .iter()
            .any(|validity| *validity == FeatureCellValidity::Valid),
        "missing volume disabled a price-only indicator"
    );
    assert!(
        rsi.validity
            .iter()
            .all(|validity| *validity != FeatureCellValidity::MissingInput)
    );
}

#[test]
fn classic_vector_ta_missing_timestamps_only_invalidates_timestamp_inputs() {
    let mut bars = fixture(240, true);
    bars.timestamp = None;
    let columns =
        compute_classic_ta_feature_columns_f64(&bars, IndicatorComputePolicy::CpuOnly, 240)
            .expect("classic/vector-ta columns without timestamps");

    let vwap = columns
        .iter()
        .find(|column| column.name == "vwap")
        .expect("VWAP");
    assert!(
        vwap.validity
            .iter()
            .all(|validity| *validity == FeatureCellValidity::MissingInput)
    );
    assert!(vwap.values.iter().all(|value| value.is_nan()));

    let obv = columns
        .iter()
        .find(|column| column.name == "obv")
        .expect("timestamp-independent OBV");
    assert!(
        obv.validity
            .iter()
            .any(|validity| *validity == FeatureCellValidity::Valid),
        "missing timestamps disabled a timestamp-independent indicator"
    );
}

#[test]
fn classic_vector_ta_preflight_placeholder_is_typed_warmup() {
    let columns = compute_classic_ta_feature_columns_f64(
        &fixture(60, true),
        IndicatorComputePolicy::CpuOnly,
        60,
    )
    .expect("short-frame classic/vector-ta columns");
    let rsi_100 = columns
        .iter()
        .find(|column| column.name == "rsi_100")
        .expect("stable-schema preflight placeholder");
    assert!(rsi_100.values.iter().all(|value| value.is_nan()));
    assert!(
        rsi_100
            .validity
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Warmup),
        "an all-NaN preflight placeholder needs the explicit warmup reason"
    );
}

#[test]
fn classic_vector_ta_rejects_noncanonical_timestamp_units() {
    let mut bars = fixture(80, true);
    bars.timestamp = bars.timestamp.map(|timestamps| {
        timestamps
            .into_iter()
            .map(|milliseconds| milliseconds * 1_000_000)
            .collect()
    });
    let error = compute_classic_ta_feature_columns_f64(&bars, IndicatorComputePolicy::CpuOnly, 80)
        .expect_err("canonical classic/vector-ta path must not infer nanoseconds");
    assert!(error.to_string().contains("millisecond"));
}

#[test]
fn footprint_missing_volume_is_invalid_but_fix_window_zero_is_valid() {
    let columns = compute_footprint_feature_columns_f64(&fixture(16, false))
        .expect("validity-aware footprint");
    let fix = columns
        .iter()
        .find(|column| column.name == "fp_fix_window")
        .expect("fix window");
    assert!(
        fix.validity
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Valid),
        "a timestamp-backed false flag is a valid mathematical zero"
    );
    assert!(fix.values.iter().all(|value| *value == 0.0));

    for column in columns
        .iter()
        .filter(|column| column.name != "fp_fix_window")
    {
        assert!(
            column
                .validity
                .iter()
                .all(|validity| *validity == FeatureCellValidity::MissingInput),
            "{} invented numeric volume data",
            column.name
        );
        assert!(column.values.iter().all(|value| value.is_nan()));
    }
}

#[test]
fn footprint_warmup_and_zero_denominator_are_not_numeric_zero() {
    let mut constant = fixture(16, true);
    constant.open.fill(1.0);
    constant.high.fill(1.0);
    constant.low.fill(1.0);
    constant.close.fill(1.0);
    constant.volume.as_mut().expect("volume").fill(100.0);
    let columns =
        compute_footprint_feature_columns_f64(&constant).expect("validity-aware footprint");

    let volume_z = columns
        .iter()
        .find(|column| column.name == "fp_volume_z")
        .expect("volume z");
    assert_eq!(volume_z.validity[0], FeatureCellValidity::Warmup);
    assert!(volume_z.values[0].is_nan());
    assert!(
        volume_z.validity[1..]
            .iter()
            .all(|validity| *validity == FeatureCellValidity::ZeroDenominator)
    );

    let delta = columns
        .iter()
        .find(|column| column.name == "fp_delta_proxy")
        .expect("delta proxy");
    assert!(
        delta
            .validity
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Valid)
    );
    assert!(
        delta.values.iter().all(|value| *value == 0.0),
        "zero signed volume over positive volume is a valid zero"
    );
}

#[test]
fn footprint_prefix_is_append_invariant() {
    let base = fixture(120, true);
    let mut extended = fixture(140, true);
    for row in 120..140 {
        extended.open[row] *= 100.0;
        extended.high[row] *= 100.0;
        extended.low[row] *= 100.0;
        extended.close[row] *= 100.0;
        extended.volume.as_mut().expect("volume")[row] *= 1.0e9;
    }

    let base_columns =
        compute_footprint_feature_columns_f64(&base).expect("base footprint columns");
    let extended_columns =
        compute_footprint_feature_columns_f64(&extended).expect("extended footprint columns");
    assert_eq!(base_columns.len(), extended_columns.len());
    for (base, extended) in base_columns.iter().zip(&extended_columns) {
        assert_eq!(base.name, extended.name);
        assert_eq!(
            base.values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            extended.values[..120]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(base.validity, extended.validity[..120]);
    }
}

#[test]
fn footprint_rejects_noncanonical_timestamp_units() {
    let mut bars = fixture(16, true);
    bars.timestamp = bars.timestamp.map(|timestamps| {
        timestamps
            .into_iter()
            .map(|milliseconds| milliseconds * 1_000_000)
            .collect()
    });
    let error = compute_footprint_feature_columns_f64(&bars)
        .expect_err("canonical production route must not infer nanoseconds");
    assert!(error.to_string().contains("millisecond"));
}

#[test]
fn session_boundaries_expose_warmup_instead_of_prefilled_zero() {
    let columns =
        compute_session_feature_columns_f64(&session_fixture(24, true)).expect("session columns");
    let london = columns
        .iter()
        .find(|column| column.name == "session_london_open_dist")
        .expect("London open distance");
    assert!(
        london.validity[..7]
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Warmup)
    );
    assert_eq!(london.validity[7], FeatureCellValidity::Valid);
    assert!(london.values[..7].iter().all(|value| value.is_nan()));

    let overlap = columns
        .iter()
        .find(|column| column.name == "session_london_ny_overlap")
        .expect("overlap flag");
    assert_eq!(overlap.validity[0], FeatureCellValidity::Valid);
    assert_eq!(overlap.values[0], 0.0, "false is a valid event flag");
    assert_eq!(overlap.values[12], 1.0);
}

#[test]
fn session_vwap_requires_real_volume_without_disabling_price_features() {
    let columns =
        compute_session_feature_columns_f64(&session_fixture(24, false)).expect("session columns");
    let london_open = columns
        .iter()
        .find(|column| column.name == "session_london_open_dist")
        .expect("London open distance");
    let london_vwap = columns
        .iter()
        .find(|column| column.name == "session_london_vwap_dist")
        .expect("London VWAP distance");
    assert_eq!(london_open.validity[7], FeatureCellValidity::Valid);
    assert_eq!(london_vwap.validity[7], FeatureCellValidity::MissingInput);
    assert!(london_vwap.values[7].is_nan());
}

#[test]
fn session_missing_timestamps_cannot_emit_numeric_calendar_features() {
    let mut bars = session_fixture(24, true);
    bars.timestamp = None;
    let columns =
        compute_session_feature_columns_f64(&bars).expect("typed missing timestamp result");
    for column in columns {
        assert!(
            column
                .validity
                .iter()
                .all(|validity| *validity == FeatureCellValidity::MissingInput),
            "{} invented calendar state without timestamps",
            column.name
        );
        assert!(column.values.iter().all(|value| value.is_nan()));
    }
}

#[test]
fn flat_daily_range_is_zero_denominator_not_neutral_midpoint() {
    let mut bars = session_fixture(4, true);
    bars.open.fill(1.0);
    bars.high.fill(1.0);
    bars.low.fill(1.0);
    bars.close.fill(1.0);
    let columns = compute_session_feature_columns_f64(&bars).expect("session columns");
    let position = columns
        .iter()
        .find(|column| column.name == "daily_position")
        .expect("daily position");
    assert_eq!(position.validity[0], FeatureCellValidity::ZeroDenominator);
    assert!(position.values[0].is_nan());
}

#[test]
fn regime_neutral_prefills_are_warmup_not_observations() {
    let columns = compute_regime_feature_columns_f64(&fixture(40, true)).expect("regime columns");
    let choppiness = columns
        .iter()
        .find(|column| column.name == "regime_choppiness")
        .expect("choppiness");
    assert!(
        choppiness.validity[..14]
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Warmup)
    );
    assert!(choppiness.values[..14].iter().all(|value| value.is_nan()));
}

#[test]
fn flat_regime_denominators_are_invalid_but_entropy_zero_is_valid() {
    let mut bars = fixture(80, true);
    bars.open.fill(1.0);
    bars.high.fill(1.0);
    bars.low.fill(1.0);
    bars.close.fill(1.0);
    let columns = compute_regime_feature_columns_f64(&bars).expect("regime columns");
    for (name, row) in [
        ("regime_vol_state", 50),
        ("regime_trend_strength", 14),
        ("regime_squeeze", 20),
        ("regime_mr_vs_momentum", 21),
        ("regime_rei", 8),
        ("regime_choppiness", 14),
        ("regime_cusum_up", 50),
    ] {
        let column = columns
            .iter()
            .find(|column| column.name == name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(
            column.validity[row],
            FeatureCellValidity::ZeroDenominator,
            "{name} treated a zero denominator as a neutral observation"
        );
        assert!(column.values[row].is_nan());
    }

    let entropy = columns
        .iter()
        .find(|column| column.name == "regime_entropy")
        .expect("entropy");
    assert_eq!(entropy.validity[30], FeatureCellValidity::Valid);
    assert_eq!(
        entropy.values[30], 0.0,
        "constant distribution entropy is zero"
    );
}

#[test]
fn regime_prefix_is_append_invariant() {
    let base = fixture(80, true);
    let mut extended = fixture(100, true);
    for row in 80..100 {
        extended.open[row] *= 50.0;
        extended.high[row] *= 50.0;
        extended.low[row] *= 50.0;
        extended.close[row] *= 50.0;
    }
    let base_columns = compute_regime_feature_columns_f64(&base).expect("base regime");
    let extended_columns = compute_regime_feature_columns_f64(&extended).expect("extended regime");
    for (base, extended) in base_columns.iter().zip(&extended_columns) {
        assert_eq!(base.name, extended.name);
        assert_eq!(
            base.values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            extended.values[..80]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(base.validity, extended.validity[..80]);
    }
}

#[test]
fn quantitative_warmup_and_zero_denominators_are_not_neutral_observations() {
    let mut bars = fixture(130, true);
    bars.open.fill(1.0);
    bars.high.fill(1.0);
    bars.low.fill(1.0);
    bars.close.fill(1.0);
    bars.volume.as_mut().expect("volume").fill(100.0);

    let columns =
        compute_quant_feature_columns_f64(&bars).expect("validity-aware quantitative columns");
    let find = |name: &str| {
        columns
            .iter()
            .find(|column| column.name == name)
            .unwrap_or_else(|| panic!("missing quantitative column {name}"))
    };

    let return_5 = find("quant_return_5");
    assert!(
        return_5.validity[..5]
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Warmup)
    );
    assert_eq!(return_5.validity[5], FeatureCellValidity::Valid);
    assert_eq!(return_5.values[5], 0.0, "flat return is a valid zero");

    let close = find("quant_close");
    assert_eq!(close.validity[0], FeatureCellValidity::Valid);
    assert_eq!(close.values[0], 1.0);

    for (name, row) in [
        ("quant_vol_ratio", 20),
        ("quant_hurst_100", 100),
        ("quant_autocorr_1", 51),
        ("quant_efficiency_ratio_10", 10),
        ("quant_skewness_30", 30),
        ("quant_kurtosis_30", 30),
        ("quant_log_volatility", 0),
        ("quant_kyle_lambda", 20),
        ("quant_body_ratio", 0),
        ("quant_upper_shadow", 0),
        ("quant_lower_shadow", 0),
        ("quant_fractal_dim", 30),
        ("quant_delta_volume", 0),
        ("quant_cum_delta_zscore", 50),
    ] {
        let column = find(name);
        assert_eq!(
            column.validity[row],
            FeatureCellValidity::ZeroDenominator,
            "{name} treated a zero denominator as a neutral observation"
        );
        assert!(column.values[row].is_nan());
    }

    let inside = find("quant_inside_bar");
    assert_eq!(inside.validity[1], FeatureCellValidity::Valid);
    assert_eq!(inside.values[1], 1.0);

    let relative_volume = find("quant_rvol_10");
    assert_eq!(relative_volume.validity[10], FeatureCellValidity::Valid);
    assert_eq!(
        relative_volume.values[10], 1.0,
        "current volume equal to a non-zero rolling average is a valid ratio"
    );
}

#[test]
fn quantitative_missing_volume_keeps_schema_but_never_fabricates_volume_features() {
    let columns = compute_quant_feature_columns_f64(&fixture(80, false))
        .expect("quantitative columns without volume");
    for name in [
        "quant_kyle_lambda",
        "quant_vpin",
        "quant_amihud_illiquidity",
        "quant_engulfing_vol",
        "quant_rvol_10",
        "quant_rvol_20",
        "quant_rvol_50",
        "quant_delta_volume",
        "quant_cum_delta_zscore",
    ] {
        let column = columns
            .iter()
            .find(|column| column.name == name)
            .unwrap_or_else(|| panic!("missing stable-schema volume column {name}"));
        assert!(
            column
                .validity
                .iter()
                .all(|validity| *validity == FeatureCellValidity::MissingInput),
            "{name} invented volume observations"
        );
        assert!(column.values.iter().all(|value| value.is_nan()));
    }

    let return_1 = columns
        .iter()
        .find(|column| column.name == "quant_return_1")
        .expect("price-only return");
    assert_eq!(return_1.validity[1], FeatureCellValidity::Valid);
}

#[test]
fn quantitative_session_proxies_fail_closed_without_a_typed_session_contract() {
    let columns =
        compute_quant_feature_columns_f64(&fixture(400, true)).expect("quantitative columns");
    for name in [
        "quant_prev_day_h_dist",
        "quant_prev_day_l_dist",
        "quant_prev_week_h_dist",
        "quant_prev_week_l_dist",
        "quant_orb_4",
        "quant_orb_8",
        "quant_orb_12",
        "quant_pivot_dist",
        "quant_r1_dist",
        "quant_r2_dist",
        "quant_s1_dist",
        "quant_s2_dist",
        "quant_cam_r3_dist",
        "quant_cam_s3_dist",
    ] {
        let column = columns
            .iter()
            .find(|column| column.name == name)
            .unwrap_or_else(|| panic!("missing session-dependent column {name}"));
        assert!(
            column
                .validity
                .iter()
                .all(|validity| *validity == FeatureCellValidity::MissingInput),
            "{name} guessed a broker session/timeframe contract"
        );
        assert!(column.values.iter().all(|value| value.is_nan()));
    }
}

#[test]
fn quantitative_rejects_noncanonical_timestamp_units() {
    let mut bars = fixture(80, true);
    bars.timestamp = bars.timestamp.map(|timestamps| {
        timestamps
            .into_iter()
            .map(|milliseconds| milliseconds * 1_000_000)
            .collect()
    });
    let error = compute_quant_feature_columns_f64(&bars)
        .expect_err("canonical quantitative path must not infer nanoseconds");
    assert!(error.to_string().contains("millisecond"));
}

#[test]
fn quantitative_prefix_is_append_invariant() {
    let base = fixture(130, true);
    let mut extended = fixture(150, true);
    for row in 130..150 {
        extended.open[row] *= 100.0;
        extended.high[row] *= 100.0;
        extended.low[row] *= 100.0;
        extended.close[row] *= 100.0;
        extended.volume.as_mut().expect("volume")[row] *= 1.0e9;
    }

    let base_columns = compute_quant_feature_columns_f64(&base).expect("base quantitative");
    let extended_columns =
        compute_quant_feature_columns_f64(&extended).expect("extended quantitative");
    assert_eq!(base_columns.len(), extended_columns.len());
    for (base, extended) in base_columns.iter().zip(&extended_columns) {
        assert_eq!(base.name, extended.name);
        assert_eq!(
            base.values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            extended.values[..130]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{} changed when future rows were appended",
            base.name
        );
        assert_eq!(base.validity, extended.validity[..130]);
    }
}

#[test]
fn smc_missing_timestamps_cannot_invent_calendar_or_session_state() {
    let mut bars = fixture(100, true);
    bars.timestamp = None;
    let columns = compute_smc_feature_columns_f64(&bars).expect("SMC without timestamps");
    for name in [
        "smc_killzone",
        "smc_asian_range",
        "smc_silver_bullet",
        "smc_judas_swing",
        "smc_nwog",
        "smc_ndog",
        "smc_ict_macro",
    ] {
        let column = columns
            .iter()
            .find(|column| column.name == name)
            .unwrap_or_else(|| panic!("missing SMC calendar column {name}"));
        assert!(
            column
                .validity
                .iter()
                .all(|validity| *validity == FeatureCellValidity::MissingInput),
            "{name} invented a UTC/broker-session observation"
        );
        assert!(column.values.iter().all(|value| value.is_nan()));
    }

    let fvg = columns
        .iter()
        .find(|column| column.name == "smc_fvg")
        .expect("price-only FVG");
    assert_eq!(fvg.validity[2], FeatureCellValidity::Valid);
}

#[test]
fn smc_warmup_denominators_and_absent_magnets_are_explicit() {
    let mut bars = fixture(120, true);
    bars.open.fill(1.0);
    bars.high.fill(1.0);
    bars.low.fill(1.0);
    bars.close.fill(1.0);
    let columns = compute_smc_feature_columns_f64(&bars).expect("flat SMC columns");
    let find = |name: &str| {
        columns
            .iter()
            .find(|column| column.name == name)
            .unwrap_or_else(|| panic!("missing SMC column {name}"))
    };

    let fvg = find("smc_fvg");
    assert!(
        fvg.validity[..2]
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Warmup)
    );
    assert_eq!(fvg.validity[2], FeatureCellValidity::Valid);
    assert_eq!(fvg.values[2], 0.0, "no three-bar gap is a valid false flag");

    for (name, row) in [
        ("smc_pd_array", 40),
        ("smc_fib_500", 40),
        ("smc_displacement", 20),
        ("smc_rejection_block", 0),
        ("smc_trend_bias", 50),
        ("smc_propulsion_block", 20),
    ] {
        let column = find(name);
        assert_eq!(
            column.validity[row],
            FeatureCellValidity::ZeroDenominator,
            "{name} converted an undefined ratio into a neutral value"
        );
        assert!(column.values[row].is_nan());
    }

    let dealing_range = find("smc_dealing_range_width");
    assert_eq!(dealing_range.validity[40], FeatureCellValidity::Valid);
    assert_eq!(dealing_range.values[40], 0.0);

    for name in ["smc_fvg_magnet_dist", "smc_fvg_magnet_age"] {
        let column = find(name);
        assert!(
            column
                .validity
                .iter()
                .all(|validity| *validity == FeatureCellValidity::AlignmentMissing)
        );
        assert!(column.values.iter().all(|value| value.is_nan()));
    }
    let open_count = find("smc_fvg_open_count");
    assert!(
        open_count
            .validity
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Valid)
    );
    assert!(open_count.values.iter().all(|value| *value == 0.0));
}

#[test]
fn smc_fvg_magnet_becomes_valid_only_when_a_gap_exists() {
    let mut bars = fixture(8, true);
    bars.open = vec![1.00, 1.05, 1.20, 1.21, 1.22, 1.23, 1.24, 1.25];
    bars.close = vec![1.00, 1.06, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26];
    bars.high = vec![1.01, 1.07, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27];
    bars.low = vec![0.99, 1.04, 1.19, 1.20, 1.21, 1.22, 1.23, 1.24];

    let columns = compute_smc_feature_columns_f64(&bars).expect("SMC gap fixture");
    let find = |name: &str| {
        columns
            .iter()
            .find(|column| column.name == name)
            .unwrap_or_else(|| panic!("missing SMC column {name}"))
    };
    assert!(find("smc_fvg_open_count").values[2] > 0.0);
    assert_eq!(
        find("smc_fvg_magnet_dist").validity[2],
        FeatureCellValidity::Valid
    );
    assert_eq!(
        find("smc_fvg_magnet_age").validity[2],
        FeatureCellValidity::Valid
    );
}

#[test]
fn smc_rejects_noncanonical_timestamp_units() {
    let mut bars = fixture(80, true);
    bars.timestamp = bars.timestamp.map(|timestamps| {
        timestamps
            .into_iter()
            .map(|milliseconds| milliseconds * 1_000_000)
            .collect()
    });
    let error = compute_smc_feature_columns_f64(&bars)
        .expect_err("canonical SMC path must not infer nanoseconds");
    assert!(error.to_string().contains("millisecond"));
}

#[test]
fn smc_prefix_is_append_invariant() {
    let base = fixture(120, true);
    let mut extended = fixture(140, true);
    for row in 120..140 {
        extended.open[row] *= 100.0;
        extended.high[row] *= 100.0;
        extended.low[row] *= 100.0;
        extended.close[row] *= 100.0;
    }

    let base_columns = compute_smc_feature_columns_f64(&base).expect("base SMC");
    let extended_columns = compute_smc_feature_columns_f64(&extended).expect("extended SMC");
    assert_eq!(base_columns.len(), extended_columns.len());
    for (base, extended) in base_columns.iter().zip(&extended_columns) {
        assert_eq!(base.name, extended.name);
        assert_eq!(
            base.values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            extended.values[..120]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{} changed when future rows were appended",
            base.name
        );
        assert_eq!(base.validity, extended.validity[..120]);
    }
}
