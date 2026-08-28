use std::str::FromStr;

use neoethos_dataset_contracts::{BarTimestampConvention, CanonicalTimeframe};

#[test]
fn the_official_ctrader_timeframe_set_has_one_stable_order_and_code() {
    use CanonicalTimeframe as T;

    let expected = [
        (T::M1, "M1", 1, Some(60_000)),
        (T::M2, "M2", 2, Some(120_000)),
        (T::M3, "M3", 3, Some(180_000)),
        (T::M4, "M4", 4, Some(240_000)),
        (T::M5, "M5", 5, Some(300_000)),
        (T::M10, "M10", 6, Some(600_000)),
        (T::M15, "M15", 7, Some(900_000)),
        (T::M30, "M30", 8, Some(1_800_000)),
        (T::H1, "H1", 9, Some(3_600_000)),
        (T::H4, "H4", 10, Some(14_400_000)),
        (T::H12, "H12", 11, Some(43_200_000)),
        (T::D1, "D1", 12, None),
        (T::W1, "W1", 13, None),
        (T::MN1, "MN1", 14, None),
    ];

    assert_eq!(CanonicalTimeframe::ALL.len(), 14);
    for (index, &(timeframe, label, protocol_code, fixed_ms)) in expected.iter().enumerate() {
        assert_eq!(CanonicalTimeframe::ALL[index], timeframe);
        assert_eq!(timeframe.as_str(), label);
        assert_eq!(timeframe.ctrader_protocol_code(), protocol_code);
        assert_eq!(timeframe.fixed_duration_ms(), fixed_ms);
        assert_eq!(
            CanonicalTimeframe::from_ctrader_protocol_code(protocol_code),
            Ok(timeframe)
        );
        assert_eq!(CanonicalTimeframe::from_str(label), Ok(timeframe));
    }
}

#[test]
fn timeframe_parsing_is_strict_and_never_invents_h2_or_calendar_minutes() {
    for invalid in [
        "", "m1", " M1", "M1 ", "H2", "H3", "H6", "H8", "M6", "M12", "M20",
    ] {
        assert!(
            CanonicalTimeframe::from_str(invalid).is_err(),
            "accepted {invalid:?}"
        );
    }
    for invalid_code in [-1, 0, 15, 99] {
        assert!(CanonicalTimeframe::from_ctrader_protocol_code(invalid_code).is_err());
    }
    assert_eq!(CanonicalTimeframe::D1.fixed_duration_ms(), None);
    assert_eq!(CanonicalTimeframe::W1.fixed_duration_ms(), None);
    assert_eq!(CanonicalTimeframe::MN1.fixed_duration_ms(), None);
}

#[test]
fn bar_timestamp_convention_is_typed_and_only_open_is_canonical() {
    let cases = [
        (BarTimestampConvention::BarOpen, "bar_open", true),
        (BarTimestampConvention::BarClose, "bar_close", false),
        (BarTimestampConvention::BarEnd, "bar_end", false),
        (BarTimestampConvention::Unknown, "unknown", false),
    ];
    for (value, label, canonical) in cases {
        assert_eq!(value.as_str(), label);
        assert_eq!(value.is_canonical_bar_open(), canonical);
        assert_eq!(BarTimestampConvention::from_str(label), Ok(value));
    }
    for invalid in ["", "open", "BAR_OPEN", "bar open", " bar_open"] {
        assert!(BarTimestampConvention::from_str(invalid).is_err());
    }
}
