use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use neoethos_data::core::feature_registry::{
    ProductionFeatureProducerId, production_feature_producer_manifest_v1,
};
use neoethos_feature_contracts::{
    RelevantDependencySetV1, RelevantDependencySourceKindV1, RelevantDependencyV1,
    SemanticSourceEntryV1, SemanticSourceKindV1, SemanticSourceManifestV1, SemanticSourceSetV1,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neoethos-data must be under <workspace>/crates")
        .to_path_buf()
}

fn cargo_lock_package<'a>(lock: &'a str, package: &str, version: &str) -> Option<&'a str> {
    lock.split("[[package]]").find(|block| {
        block
            .lines()
            .any(|line| line == format!("name = \"{package}\""))
            && block
                .lines()
                .any(|line| line == format!("version = \"{version}\""))
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn embedded_semantic_sources_match_current_canonical_source_bytes() {
    let root = workspace_root();
    let expected_primary = [
        (
            ProductionFeatureProducerId::SmartMoneyConcept,
            "crates/neoethos-data/src/core/smc.rs",
        ),
        (
            ProductionFeatureProducerId::ClassicVectorTa,
            "crates/neoethos-data/src/core/hpc_ta.rs",
        ),
        (
            ProductionFeatureProducerId::Quantitative,
            "crates/neoethos-data/src/core/quant_features.rs",
        ),
        (
            ProductionFeatureProducerId::Session,
            "crates/neoethos-data/src/core/session_features.rs",
        ),
        (
            ProductionFeatureProducerId::Regime,
            "crates/neoethos-data/src/core/regime_detection.rs",
        ),
        (
            ProductionFeatureProducerId::Footprint,
            "crates/neoethos-data/src/core/footprint_features.rs",
        ),
    ];

    for row in production_feature_producer_manifest_v1().expect("embedded producer manifest") {
        let entries = row.semantic_sources().entries();
        assert!(
            entries
                .windows(2)
                .all(|pair| pair[0].path().as_bytes() < pair[1].path().as_bytes()),
            "{:?} semantic paths must be unique and byte-sorted",
            row.producer()
        );
        for entry in entries {
            let path = entry.path();
            let current = fs::read(root.join(path))
                .unwrap_or_else(|error| panic!("cannot read semantic source `{path}`: {error}"));
            let rebuilt =
                SemanticSourceEntryV1::from_bytes(path, SemanticSourceKindV1::Utf8Text, &current)
                    .expect("current source is canonicalizable UTF-8");
            assert_eq!(
                entry.payload_hash(),
                rebuilt.payload_hash(),
                "embedded source `{path}` is stale"
            );
        }
        let primary = expected_primary
            .iter()
            .find(|(producer, _)| producer == &row.producer())
            .expect("every producer needs an expected primary source")
            .1;
        assert!(
            entries.iter().any(|entry| entry.path() == primary),
            "{:?} omits its production implementation `{primary}`",
            row.producer()
        );
    }
}

#[test]
fn smc_semantic_v3_binds_the_shared_exact_logarithm_authority() {
    let row = production_feature_producer_manifest_v1()
        .expect("embedded producer manifest")
        .into_iter()
        .find(|row| row.producer() == ProductionFeatureProducerId::SmartMoneyConcept)
        .expect("SMC producer row");
    assert_eq!(row.semantic_version(), 3);
    assert!(
        row.semantic_sources()
            .entries()
            .iter()
            .any(|entry| { entry.path() == "crates/neoethos-data/src/core/smc_log1p_exact_v1.rs" })
    );
}

#[test]
fn regime_semantic_v3_binds_the_source_sealed_exact_logarithm_authority() {
    let row = production_feature_producer_manifest_v1()
        .expect("embedded producer manifest")
        .into_iter()
        .find(|row| row.producer() == ProductionFeatureProducerId::Regime)
        .expect("Regime producer row");
    assert_eq!(row.semantic_version(), 3);
    assert!(
        row.semantic_sources().entries().iter().any(|entry| {
            entry.path() == "crates/neoethos-data/src/core/regime_exact_math_v1.rs"
        })
    );
}

#[test]
fn relevant_dependencies_match_the_locked_graph() {
    let root = workspace_root();
    let lock = fs::read_to_string(root.join("Cargo.lock")).expect("workspace Cargo.lock");

    for row in production_feature_producer_manifest_v1().expect("embedded producer manifest") {
        let mut packages = HashSet::new();
        for dependency in row.relevant_dependencies().entries() {
            assert!(
                packages.insert(dependency.package_name()),
                "{:?} repeats dependency {}",
                row.producer(),
                dependency.package_name()
            );
            assert!(
                dependency
                    .enabled_features()
                    .windows(2)
                    .all(|pair| pair[0].as_bytes() < pair[1].as_bytes()),
                "{} features must be unique and byte-sorted",
                dependency.package_name()
            );
            let block = cargo_lock_package(
                &lock,
                dependency.package_name(),
                dependency.resolved_version(),
            )
            .unwrap_or_else(|| {
                panic!(
                    "{:?} dependency {} {} is not locked",
                    row.producer(),
                    dependency.package_name(),
                    dependency.resolved_version()
                )
            });

            match dependency.source_kind() {
                RelevantDependencySourceKindV1::Registry => {
                    assert!(
                        block.contains(&format!(
                            "source = \"registry+{}\"",
                            dependency.source_identity()
                        )),
                        "{} source identity differs from Cargo.lock",
                        dependency.package_name()
                    );
                    assert!(
                        block.contains(&format!(
                            "checksum = \"{}\"",
                            hex(dependency.checksum_or_source_manifest_hash())
                        )),
                        "{} checksum differs from Cargo.lock",
                        dependency.package_name()
                    );
                }
                RelevantDependencySourceKindV1::RepositoryPath => {
                    assert!(
                        root.join(dependency.source_identity()).is_dir(),
                        "{} repository source path is missing",
                        dependency.package_name()
                    );
                    assert!(
                        !block.lines().any(|line| line.starts_with("source = ")),
                        "{} is a path dependency but Cargo.lock records an external source",
                        dependency.package_name()
                    );
                    assert_ne!(
                        dependency.checksum_or_source_manifest_hash(),
                        &[0; 32],
                        "{} path dependency has an unknown semantic source hash",
                        dependency.package_name()
                    );
                }
                RelevantDependencySourceKindV1::Git => {
                    panic!(
                        "{} unexpectedly became a Git dependency without an exact lock assertion",
                        dependency.package_name()
                    );
                }
            }
        }
    }
}

#[test]
fn source_and_dependency_mutations_change_only_the_declaring_payload() {
    let root = workspace_root();
    for row in production_feature_producer_manifest_v1().expect("embedded producer manifest") {
        let baseline = row.semantic_source_set().identity();
        for changed_path in row
            .semantic_sources()
            .entries()
            .iter()
            .map(|entry| entry.path())
        {
            let entries = row
                .semantic_sources()
                .entries()
                .iter()
                .map(|entry| {
                    let mut bytes = fs::read(root.join(entry.path())).expect("declared source");
                    if entry.path() == changed_path {
                        bytes.extend_from_slice(b"\n// semantic mutation fixture\n");
                    }
                    SemanticSourceEntryV1::from_bytes(
                        entry.path(),
                        SemanticSourceKindV1::Utf8Text,
                        &bytes,
                    )
                    .expect("mutated source entry")
                })
                .collect::<Vec<_>>();
            let sources = SemanticSourceManifestV1::new(entries).expect("mutated source manifest");
            let changed = SemanticSourceSetV1::new(sources, row.relevant_dependencies().clone());
            assert_ne!(
                baseline,
                changed.identity(),
                "changing declared source `{changed_path}` did not change {:?}",
                row.producer()
            );
        }

        for changed_dependency in row.relevant_dependencies().entries() {
            let dependencies = row
                .relevant_dependencies()
                .entries()
                .iter()
                .map(|dependency| {
                    let version = if dependency.package_name() == changed_dependency.package_name()
                    {
                        "999.999.999-test"
                    } else {
                        dependency.resolved_version()
                    };
                    match dependency.source_kind() {
                        RelevantDependencySourceKindV1::Registry => RelevantDependencyV1::registry(
                            dependency.package_name(),
                            version,
                            dependency.source_identity(),
                            *dependency.checksum_or_source_manifest_hash(),
                            dependency.enabled_features().to_vec(),
                        ),
                        RelevantDependencySourceKindV1::RepositoryPath => {
                            RelevantDependencyV1::repository_path(
                                dependency.package_name(),
                                version,
                                dependency.source_identity(),
                                *dependency.checksum_or_source_manifest_hash(),
                                dependency.enabled_features().to_vec(),
                            )
                        }
                        RelevantDependencySourceKindV1::Git => {
                            panic!("Git dependency reconstruction needs exact URL and revision")
                        }
                    }
                    .expect("mutated dependency")
                })
                .collect::<Vec<_>>();
            let changed = SemanticSourceSetV1::new(
                row.semantic_sources().clone(),
                RelevantDependencySetV1::new(dependencies).expect("mutated dependency set"),
            );
            assert_ne!(
                baseline,
                changed.identity(),
                "changing relevant dependency {} did not change {:?}",
                changed_dependency.package_name(),
                row.producer()
            );
        }

        assert_eq!(
            baseline,
            SemanticSourceSetV1::new(
                row.semantic_sources().clone(),
                row.relevant_dependencies().clone(),
            )
            .identity(),
            "reconstructing unchanged relevant inputs changed {:?}",
            row.producer()
        );
    }
}
