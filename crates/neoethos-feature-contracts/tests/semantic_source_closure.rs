use neoethos_feature_contracts::{
    RelevantDependencySetV1, RelevantDependencyV1, SemanticSourceEntryV1, SemanticSourceKindV1,
    SemanticSourceManifestV1, SemanticSourceSetV1,
};

fn hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

#[test]
fn relevant_dependency_encoding_is_order_independent_and_portable() {
    let registry = RelevantDependencyV1::registry(
        "chrono",
        "0.4.44",
        "https://github.com/rust-lang/crates.io-index",
        hash(1),
        vec!["serde".to_owned(), "clock".to_owned()],
    )
    .expect("registry dependency");
    let git = RelevantDependencyV1::git(
        "formula-lib",
        "1.2.3",
        "https://github.com/example/formula-lib",
        "0123456789abcdef0123456789abcdef01234567",
        hash(2),
        vec!["f64".to_owned()],
    )
    .expect("git dependency");
    let path = RelevantDependencyV1::repository_path(
        "vector-ta",
        "0.2.9",
        "vendor/vector-ta-0.2.9-patched",
        hash(3),
        vec!["cuda".to_owned(), "f64".to_owned()],
    )
    .expect("repository dependency");
    let first = RelevantDependencySetV1::new(vec![registry.clone(), git.clone(), path.clone()])
        .expect("dependency set");
    let reordered =
        RelevantDependencySetV1::new(vec![path, registry, git]).expect("reordered dependency set");
    assert_eq!(
        first.identity().to_hex(),
        "965b951114d71db917e6388e245b2ef53ac416a1480b5fd6fcbf77183560c5bb"
    );
    assert_eq!(first.identity(), reordered.identity());
    assert_eq!(first.canonical_bytes(), reordered.canonical_bytes());

    let reordered_features = RelevantDependencyV1::registry(
        "chrono",
        "0.4.44",
        "https://github.com/rust-lang/crates.io-index",
        hash(1),
        vec!["clock".to_owned(), "serde".to_owned()],
    )
    .expect("feature order canonicalizes");
    assert_eq!(
        RelevantDependencySetV1::new(vec![reordered_features])
            .expect("set")
            .entries()[0],
        RelevantDependencySetV1::new(vec![
            RelevantDependencyV1::registry(
                "chrono",
                "0.4.44",
                "https://github.com/rust-lang/crates.io-index",
                hash(1),
                vec!["serde".to_owned(), "clock".to_owned()],
            )
            .expect("registry")
        ])
        .expect("set")
        .entries()[0]
    );
}

#[test]
fn dependency_ambiguity_and_machine_local_identity_fail_closed() {
    assert!(
        RelevantDependencyV1::repository_path(
            "bad",
            "1.0.0",
            "C:/Users/me/bad",
            hash(1),
            Vec::new()
        )
        .is_err()
    );
    assert!(
        RelevantDependencyV1::git(
            "bad",
            "1.0.0",
            "https://github.com/example/bad",
            "main",
            hash(1),
            Vec::new()
        )
        .is_err()
    );
    assert!(
        RelevantDependencyV1::registry(
            "dup-feature",
            "1.0.0",
            "https://github.com/rust-lang/crates.io-index",
            hash(1),
            vec!["serde".to_owned(), "serde".to_owned()]
        )
        .is_err()
    );

    let duplicate = RelevantDependencyV1::registry(
        "chrono",
        "0.4.44",
        "https://github.com/rust-lang/crates.io-index",
        hash(1),
        Vec::new(),
    )
    .expect("dependency");
    assert!(
        RelevantDependencySetV1::new(vec![duplicate.clone(), duplicate]).is_err(),
        "duplicate package/source identity was accepted"
    );
}

#[test]
fn only_reachable_source_or_dependency_changes_the_semantic_source_set() {
    let source = |payload: &[u8]| {
        SemanticSourceManifestV1::new(vec![
            SemanticSourceEntryV1::from_bytes(
                "src/formula.rs",
                SemanticSourceKindV1::Utf8Text,
                payload,
            )
            .expect("source"),
        ])
        .expect("manifest")
    };
    let chrono = |checksum| {
        RelevantDependencySetV1::new(vec![
            RelevantDependencyV1::registry(
                "chrono",
                "0.4.44",
                "https://github.com/rust-lang/crates.io-index",
                checksum,
                vec!["clock".to_owned()],
            )
            .expect("chrono"),
        ])
        .expect("dependency set")
    };

    let baseline = SemanticSourceSetV1::new(source(b"formula\n"), chrono(hash(1)));
    let same_with_unrelated_dependency_absent =
        SemanticSourceSetV1::new(source(b"formula\r\n"), chrono(hash(1)));
    let changed_source = SemanticSourceSetV1::new(source(b"repaired formula\n"), chrono(hash(1)));
    let changed_dependency = SemanticSourceSetV1::new(source(b"formula\n"), chrono(hash(2)));

    assert_eq!(
        baseline.identity(),
        same_with_unrelated_dependency_absent.identity()
    );
    assert_ne!(baseline.identity(), changed_source.identity());
    assert_ne!(baseline.identity(), changed_dependency.identity());
}
