use neoethos_core::{CANONICAL_TIMEFRAMES, CanonicalTimeframe};

#[test]
fn core_reexports_the_exact_leaf_timeframe_type_and_official_set() {
    assert_eq!(CANONICAL_TIMEFRAMES.len(), CanonicalTimeframe::ALL.len());
    for (index, timeframe) in CanonicalTimeframe::ALL.into_iter().enumerate() {
        let leaf: neoethos_dataset_contracts::CanonicalTimeframe = timeframe;
        assert_eq!(leaf.as_str(), CANONICAL_TIMEFRAMES[index]);
        assert_eq!(
            leaf.ctrader_protocol_code(),
            i32::try_from(index + 1).unwrap()
        );
        assert_eq!(
            CanonicalTimeframe::from_ctrader_protocol_code(leaf.ctrader_protocol_code())
                .expect("official protocol code"),
            timeframe
        );
        assert_eq!(
            timeframe.as_str().parse::<CanonicalTimeframe>().unwrap(),
            timeframe
        );
    }

    for invalid in ["", "H2", "H3", "H6", "H8", "M6", "M12", "M20"] {
        assert!(
            invalid.parse::<CanonicalTimeframe>().is_err(),
            "accepted {invalid}"
        );
    }
}
