use neoethos_dataset_contracts::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalDatasetScope,
    CanonicalTimeframe,
};

fn external(symbol: &str) -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::external(
        "github-public-snapshot",
        symbol,
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid external identity")
}

#[test]
fn external_identity_has_stable_canonical_bytes_and_reversible_safe_path() {
    let identity = external("EUR/USD.pro");
    assert!(!identity.is_broker_real());
    assert_eq!(identity.symbol_name(), "EUR/USD.pro");
    assert_eq!(identity.timeframe(), CanonicalTimeframe::M5);
    assert_eq!(
        identity.bar_timestamp_convention(),
        BarTimestampConvention::BarOpen
    );

    let expected_bytes = [
        b"neoethos.canonical-dataset-identity.v1\0".as_slice(),
        &[0, 1],
        &[1],
        &(22_u32.to_be_bytes()),
        b"github-public-snapshot",
        &(11_u32.to_be_bytes()),
        b"EUR/USD.pro",
        &[5],
        &[1],
    ]
    .concat();
    assert_eq!(identity.canonical_bytes(), expected_bytes);

    let component = identity.to_path_component();
    assert!(component.starts_with("d1-"));
    assert!(component.len() <= 240);
    assert!(
        component
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    );
    assert!(!component.contains('/'));
    assert!(!component.contains('\\'));
    assert!(!component.contains(':'));
    assert_eq!(
        CanonicalDatasetIdentity::from_path_component(&component),
        Ok(identity)
    );
}

#[test]
fn broker_identity_binds_environment_server_account_symbol_id_name_and_timeframe() {
    let identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Live,
        "live.ctrader.example:5035",
        712_345,
        14,
        "XAUUSD.r",
        CanonicalTimeframe::M10,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid broker identity");

    assert!(identity.is_broker_real());
    assert_eq!(
        identity.scope(),
        &CanonicalDatasetScope::CTrader {
            environment: CTraderEnvironment::Live,
            server: "live.ctrader.example:5035".to_owned(),
            account_id: 712_345,
            symbol_id: 14,
        }
    );
    assert_eq!(
        CanonicalDatasetIdentity::from_path_component(&identity.to_path_component()),
        Ok(identity)
    );
}

#[test]
fn punctuation_unicode_case_and_windows_names_round_trip_without_collision() {
    let names = [
        "EUR/USD",
        "EURUSD",
        "EUR.USD",
        "EUR:USD",
        "EUR\\USD",
        ".",
        "..",
        "CON",
        "con",
        "NUL.txt",
        "eurusd",
        "EURUSD.pro",
        "EURUSD.r",
        "ΕΥΡΩUSD",
        "é",
        "e\u{301}",
    ];
    let mut components = std::collections::BTreeSet::new();
    for name in names {
        let identity = external(name);
        let component = identity.to_path_component();
        assert!(
            components.insert(component.clone()),
            "collision for {name:?}"
        );
        assert_eq!(
            CanonicalDatasetIdentity::from_path_component(&component),
            Ok(identity)
        );
    }
}

#[test]
fn malformed_or_noncanonical_components_and_unsafe_identity_text_fail_closed() {
    for invalid_text in ["", " ", "EUR\0USD", "EUR\nUSD", "\u{7f}"] {
        assert!(
            CanonicalDatasetIdentity::external(
                "external",
                invalid_text,
                CanonicalTimeframe::M1,
                BarTimestampConvention::BarOpen,
            )
            .is_err()
        );
    }
    assert!(
        CanonicalDatasetIdentity::external(
            "x".repeat(300),
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .is_err()
    );

    let valid = external("EURUSD").to_path_component();
    for invalid in [
        "",
        ".",
        "..",
        "d1-",
        "d2-0000",
        "d1-====",
        "d1-ABCDEF",
        "d1-000/111",
        "d1-000\\111",
        "d1-000:111",
    ] {
        assert!(
            CanonicalDatasetIdentity::from_path_component(invalid).is_err(),
            "accepted {invalid:?}"
        );
    }
    assert!(CanonicalDatasetIdentity::from_path_component(&(valid.clone() + "0")).is_err());
    assert!(CanonicalDatasetIdentity::from_path_component(&valid.to_ascii_uppercase()).is_err());
}

#[test]
fn non_open_convention_cannot_become_a_canonical_dataset_identity() {
    for convention in [
        BarTimestampConvention::BarClose,
        BarTimestampConvention::BarEnd,
        BarTimestampConvention::Unknown,
    ] {
        assert!(
            CanonicalDatasetIdentity::external(
                "external",
                "EURUSD",
                CanonicalTimeframe::M5,
                convention,
            )
            .is_err()
        );
    }
}
