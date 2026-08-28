use neoethos_data::core::feature_registry::{
    ProductionFeatureProducerId, production_feature_producer_manifest_v1,
};
use neoethos_feature_contracts::{
    RelevantDependencySetV1, SemanticSourceManifestV1, SemanticSourceSetV1,
};

fn require_shared_contract_types(
    _: &SemanticSourceManifestV1,
    _: &RelevantDependencySetV1,
    _: &SemanticSourceSetV1,
) {
}

#[test]
fn every_production_producer_uses_the_shared_canonical_contract_leaf() {
    let rows = production_feature_producer_manifest_v1().expect("embedded producer manifest");
    assert_eq!(rows.len(), 6);

    for row in rows {
        require_shared_contract_types(
            row.semantic_sources(),
            row.relevant_dependencies(),
            row.semantic_source_set(),
        );
        assert_eq!(
            row.semantic_source_set().sources().identity(),
            row.semantic_sources().identity()
        );
        assert_eq!(
            row.semantic_source_set().dependencies().identity(),
            row.relevant_dependencies().identity()
        );
    }

    assert!(rows.iter().any(|row| {
        row.producer() == ProductionFeatureProducerId::Footprint
            && !row.semantic_sources().entries().is_empty()
    }));
}
