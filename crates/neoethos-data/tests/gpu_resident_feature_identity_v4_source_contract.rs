use std::fs;
use std::path::PathBuf;

fn workspace_file(path: &str) -> String {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest.join(path)).expect("workspace source must be readable")
}

fn recipe_source() -> String {
    workspace_file("src/core/gpu_resident_feature_recipe_v4.rs")
}

fn store_source() -> String {
    workspace_file("src/core/gpu_resident_feature_store_v3.rs")
}

#[test]
fn final_identity_is_not_minted_before_runtime_normalization_evidence() {
    let store = store_source();
    assert!(
        !store.contains("fn final_identity_hashes("),
        "pre-runtime final_identity_hashes authority must be removed"
    );
    let seal = store
        .split_once("pub(crate) fn seal_gpu_resident_feature_store_v3(")
        .expect("final resident seal must exist")
        .1;
    let fit = seal
        .find("validate_runtime_evidence(&evidence)")
        .expect("normalization evidence must be validated");
    let finalize = seal
        .find("finalize_after_normalization_v4(normalization_fit_sha256)")
        .expect("feature identity must finalize from the validated fit digest");
    assert!(
        fit < finalize,
        "fit evidence must precede identity finalization"
    );
}

#[test]
fn recipe_carries_typed_plan_templates_and_data_derived_admission_hashes() {
    let recipe = recipe_source();
    for token in [
        "struct ResidentFeaturePlanRouteTemplateV4",
        "typed_parameters: Vec<ResidentCanonicalParameterV4>",
        "route_domain: String",
        "semantic_version: u32",
        "implementation_sha256: [u8; 32]",
        "struct ResidentFeatureIdentityTemplateV4",
        "dataset_recipe_sha256: [u8; 32]",
        "feature_plan_schema_sha256: [u8; 32]",
        "route_plan_sha256: [u8; 32]",
        "neoethos.data.resident-dataset-recipe.v4",
        "neoethos.data.resident-feature-plan-schema.v4",
        "neoethos.data.resident-route-plan.v4",
    ] {
        assert!(
            recipe.contains(token),
            "missing identity-template token {token}"
        );
    }
}

#[test]
fn complete_materialized_source_owner_moves_through_plan_admission_and_seal() {
    let recipe = recipe_source();
    let store = store_source();
    assert!(
        recipe.contains("sources: MaterializedPinnedResidentCanonicalSourcesV1"),
        "recipe identity must own all materialized pinned generations"
    );
    for (owner, field) in [
        (
            "ResolvedGpuOnlyFeatureMaterializationPlanV3",
            "feature_identity: ResidentFeatureIdentityTemplateV4",
        ),
        (
            "GpuOnlyFeatureRecipePreflightV3",
            "plan: ResolvedGpuOnlyFeatureMaterializationPlanV3",
        ),
        (
            "GpuOnlyFeatureMaterializationAdmissionV3",
            "feature_identity: ResidentFeatureIdentityTemplateV4",
        ),
        (
            "GpuOnlyFeatureMaterializationSealTokenV3",
            "feature_identity: ResidentFeatureIdentityTemplateV4",
        ),
        (
            "SealedGpuResidentFeatureStoreV3",
            "resident_sources: MaterializedPinnedResidentCanonicalSourcesV1",
        ),
    ] {
        let body = store
            .split_once(&format!("struct {owner}"))
            .unwrap_or_else(|| panic!("missing {owner}"))
            .1
            .split_once('}')
            .map(|(body, _)| body)
            .expect("owner body must close");
        assert!(
            body.contains(field),
            "{owner} is missing move-only field {field}"
        );
    }
    assert!(
        store.contains("feature_identity: self.feature_identity")
            && store.contains("feature_identity,\n        ..\n    } = plan"),
        "bind/begin must move the same identity continuation without cloning"
    );
}

#[test]
fn enabled_and_disabled_normalization_build_distinct_honest_plan_topologies() {
    let recipe = recipe_source();
    for token in [
        "if self.normalization_enabled",
        "FeatureOperationTagV1::Normalization",
        "Some(normalization_fit_sha256)",
        "canonical_disabled_normalization_fit_sha256_v4()",
        "normalization node is forbidden when disabled",
    ] {
        assert!(
            recipe.contains(token),
            "missing normalization topology token {token}"
        );
    }
}

#[test]
fn sealed_store_retains_exact_plan_provenance_and_source_leases() {
    let store = store_source();
    let sealed = store
        .split_once("pub struct SealedGpuResidentFeatureStoreV3")
        .expect("sealed resident store must exist")
        .1
        .split_once('}')
        .expect("sealed resident store body must close")
        .0;
    for field in [
        "feature_plan: FeaturePlanV1",
        "source_provenance: DatasetFeatureArtifactProvenanceV1",
        "resident_sources: MaterializedPinnedResidentCanonicalSourcesV1",
    ] {
        assert!(sealed.contains(field), "sealed store is missing {field}");
    }
    assert!(
        store.contains("*feature_plan.identity().as_bytes()")
            && store.contains("*source_provenance.identity().as_bytes()"),
        "only contract-derived plan/provenance identities may enter the low-level seal"
    );
}
