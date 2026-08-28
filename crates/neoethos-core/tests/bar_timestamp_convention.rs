use neoethos_core::{BarTimestampConvention, CanonicalTimeframe};
use neoethos_dataset_contracts::CanonicalDatasetIdentity;

#[test]
fn canonical_identity_accepts_only_explicit_bar_open_timestamps() {
    let valid = CanonicalDatasetIdentity::external(
        "fixture",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("explicit bar-open identity");
    assert_eq!(
        valid.bar_timestamp_convention(),
        BarTimestampConvention::BarOpen
    );

    for convention in [
        BarTimestampConvention::BarClose,
        BarTimestampConvention::BarEnd,
        BarTimestampConvention::Unknown,
    ] {
        assert!(
            CanonicalDatasetIdentity::external(
                "fixture",
                "EURUSD",
                CanonicalTimeframe::M1,
                convention,
            )
            .is_err(),
            "accepted noncanonical convention {convention}"
        );
    }
}
