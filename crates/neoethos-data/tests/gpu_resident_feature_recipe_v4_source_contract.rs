use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-data"))
}

fn read_or_empty(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join(relative)).unwrap_or_default()
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing recipe-v4 section {start:?}"))
        .1
        .split_once(end)
        .unwrap_or_else(|| panic!("missing recipe-v4 terminator {end:?}"))
        .0
}

fn require_in_order(source: &str, tokens: &[&str]) {
    let mut cursor = 0;
    for token in tokens {
        let relative = source[cursor..]
            .find(token)
            .unwrap_or_else(|| panic!("missing ordered recipe-v4 token {token:?}"));
        cursor += relative + token.len();
    }
}

fn assert_schema_and_transform_order(source: &str) {
    let schema = source
        .split_once("RESIDENT_COLUMN_SCHEMA_ORDER_V4")
        .expect("explicit schema-v4 order")
        .1
        .split_once("];")
        .expect("schema-v4 order terminator")
        .0;
    require_in_order(
        schema,
        &[
            "ResidentFeatureProducerV3::Smc",
            "ResidentFeatureProducerV3::ClassicTa",
            "ResidentFeatureProducerV3::Quant",
            "ResidentFeatureProducerV3::Session",
            "ResidentFeatureProducerV3::Regime",
            "ResidentFeatureProducerV3::Footprint",
            "ResidentFeatureProducerV3::HigherTimeframeAlignment",
        ],
    );
    assert_eq!(schema.matches("ResidentFeatureProducerV3::").count(), 7);

    let transforms = source
        .split_once("RESIDENT_TRANSFORM_ORDER_V4")
        .expect("separate transform order")
        .1
        .split_once("];")
        .expect("transform order terminator")
        .0;
    require_in_order(
        transforms,
        &[
            "ResidentFeatureProducerV3::RobustNormalization",
            "ResidentFeatureProducerV3::CanonicalContentSha256",
            "ResidentFeatureProducerV3::FeatureMajorToBarMajor",
        ],
    );
    assert_eq!(transforms.matches("ResidentFeatureProducerV3::").count(), 3);
}

fn assert_guessed_zero_hash_is_rejected(source: &str) {
    let compact = compact(source);
    assert!(
        compact.contains(
            "ResidentCanonicalParameterValueV4::Hash(hash)ifhash.iter().all(|byte|*byte==0)=>"
        ),
        "typed parameter authority must reject a guessed all-zero hash"
    );
    assert!(
        source.contains("fn guessed_zero_hash_parameter_is_rejected_before_draft"),
        "missing executed guessed-zero-hash fixture"
    );
}

fn assert_move_only_private_type(source: &str, declaration: &str, type_name: &str) {
    let declaration_index = source
        .find(declaration)
        .unwrap_or_else(|| panic!("missing move-only declaration {declaration:?}"));
    let derive_start = source[..declaration_index]
        .rfind("#[derive(")
        .unwrap_or_else(|| panic!("missing derive list for {type_name}"));
    let derives = &source[derive_start..declaration_index];
    assert!(
        !derives.contains("Clone") && !derives.contains("Copy"),
        "{type_name} gained Clone/Copy"
    );
    assert!(
        !source.contains(&format!("impl Clone for {type_name}"))
            && !source.contains(&format!("impl Copy for {type_name}")),
        "{type_name} gained a manual Clone/Copy implementation"
    );
    let body = section(source, declaration, "\n}");
    assert!(
        !body
            .lines()
            .any(|line| line.trim_start().starts_with("pub ")
                || line.trim_start().starts_with("pub(")),
        "{type_name} exposes a field"
    );
}

#[test]
fn schema_order_is_cpu_authority_and_transforms_are_separate() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    let modules = read_or_empty("src/core/mod.rs");
    assert!(
        modules.contains("pub(crate) mod gpu_resident_feature_recipe_v4;"),
        "Data must register the crate-private recipe-v4 authority"
    );
    assert_schema_and_transform_order(&source);
}

#[test]
fn local_drafts_are_move_only_and_cannot_mint_identity_material() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    for required in [
        "pub(crate) struct ResidentProducerDraftV4",
        "pub(crate) struct ResidentRouteDraftV4",
        "pub(crate) struct ResidentProducerBatchDraftV4",
        "pub(crate) enum ResidentCanonicalParameterValueV4",
        "typed_parameters",
        "derive_parameter_tuple_sha256_v4",
        "derive_route_receipt_sha256_v4",
    ] {
        assert!(
            source.contains(required),
            "missing typed draft token {required:?}"
        );
    }
    for (declaration, type_name) in [
        (
            "pub(crate) struct ResidentProducerDraftV4 {",
            "ResidentProducerDraftV4",
        ),
        (
            "pub(crate) struct ResidentRouteDraftV4 {",
            "ResidentRouteDraftV4",
        ),
        (
            "pub(crate) struct ResidentProducerBatchDraftV4 {",
            "ResidentProducerBatchDraftV4",
        ),
        (
            "pub(crate) struct ResidentTransformCapabilityDraftV4 {",
            "ResidentTransformCapabilityDraftV4",
        ),
    ] {
        assert_move_only_private_type(&source, declaration, type_name);
    }
    let route_draft = section(&source, "pub(crate) struct ResidentRouteDraftV4 {", "\n}");
    for forbidden in [
        "parameter_tuple_sha256: [u8; 32]",
        "ordinal: u64",
        "route_receipt_sha256: [u8; 32]",
        "route_id: String",
    ] {
        assert!(
            !route_draft.contains(forbidden),
            "local recipe draft can mint authority through {forbidden:?}"
        );
    }
    for forbidden in ["pub fn new(", "from_hash", "from_bytes"] {
        assert!(
            !source.contains(forbidden),
            "local recipe draft exposes {forbidden:?}"
        );
    }
}

#[test]
fn batches_and_feature_names_are_checked_before_global_seal() {
    let source = compact(&read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs"));
    for required in [
        "column_count==0",
        "column_count>MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V4",
        "local_first_column!=next_local_column",
        "DuplicateFeatureName",
        "ProducerOrderMismatch",
        "MissingColumnProducers",
        "RESIDENT_COLUMN_SCHEMA_ORDER_V4.len()",
        "capabilities_by_producer",
    ] {
        assert!(
            source.contains(required),
            "missing fail-closed recipe check {required:?}"
        );
    }
}

#[test]
fn mutation_audit_rejects_schema_reordering_and_transform_pseudo_columns() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    for mutated in [
        source.replacen(
            "ResidentFeatureProducerV3::Smc,\n    ResidentFeatureProducerV3::ClassicTa,",
            "ResidentFeatureProducerV3::ClassicTa,\n    ResidentFeatureProducerV3::Smc,",
            1,
        ),
        source.replacen(
            "ResidentFeatureProducerV3::HigherTimeframeAlignment,",
            "ResidentFeatureProducerV3::RobustNormalization,",
            1,
        ),
    ] {
        assert!(
            std::panic::catch_unwind(|| assert_schema_and_transform_order(&mutated)).is_err(),
            "source contract accepted a mutated schema authority"
        );
    }
}

#[test]
fn mutation_audit_rejects_removing_the_guessed_zero_hash_guard() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    assert_guessed_zero_hash_is_rejected(&source);
    let mutated = source.replacen(
        "ResidentCanonicalParameterValueV4::Hash(hash) if hash.iter().all(|byte| *byte == 0) =>",
        "ResidentCanonicalParameterValueV4::Hash(_) =>",
        1,
    );
    assert_ne!(source, mutated, "zero-hash mutation fixture did not apply");
    assert!(
        std::panic::catch_unwind(|| assert_guessed_zero_hash_is_rejected(&mutated)).is_err(),
        "source contract accepted removal of the guessed-zero-hash guard"
    );
}

#[test]
fn route_id_is_bound_to_the_complete_route_receipt() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    for required in [
        "fn derive_route_id_v4(",
        "sha256_hex_v4(route_receipt_sha256)",
        "derive_route_id_v4(global_column, producer, &route.feature_name, route_receipt)",
        "fn route_identity_changes_when_full_semantic_fragment_changes",
    ] {
        assert!(
            source.contains(required),
            "route id is not bound to full semantic identity through {required:?}"
        );
    }
    let old_partial_id = "format!(\n                    \"neoethos.data.resident-feature-schema.v4:{}:{global_column}:{}\"";
    assert!(
        !source.contains(old_partial_id),
        "route id still aliases distinct semantic fragments"
    );
}

#[test]
fn capability_manifest_is_reconstructed_in_contract_order_not_schema_order() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    for required in [
        "pub(crate) struct ResidentTransformCapabilityDraftV4",
        "ResidentFeatureProducerV3::ALL",
        "ResidentProducerCapabilityManifestV3::seal(ordered_capabilities)",
        "capability_manifest: ResidentProducerCapabilityManifestV3",
        "fn ten_entry_capability_manifest_uses_contract_order",
        "fn missing_or_mislabelled_transform_capability_fails_closed",
    ] {
        assert!(
            source.contains(required),
            "missing complete capability-manifest authority {required:?}"
        );
    }
    let seal_signature = section(
        &source,
        "pub(crate) fn seal(",
        ") -> Result<SealedResidentColumnSchemaV4",
    );
    assert!(
        seal_signature.contains("transform_capabilities: ResidentTransformCapabilityDraftV4"),
        "seven column drafts can seal without the three transform capabilities"
    );
}

#[test]
fn whitespace_only_semantic_identity_fails_closed() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    for required in [
        "fn semantic_text_is_blank(value: &str) -> bool",
        "value.trim().is_empty()",
        "semantic_text_is_blank(&name)",
        "ResidentCanonicalParameterValueV4::Text(text) if semantic_text_is_blank(text)",
        "semantic_text_is_blank(&feature_name)",
        "semantic_text_is_blank(route_domain)",
        "is_some_and(semantic_text_is_blank)",
        "fn whitespace_only_semantic_fields_fail_closed",
    ] {
        assert!(
            source.contains(required),
            "whitespace-only identity can bypass {required:?}"
        );
    }
    let mutated = source.replacen("value.trim().is_empty()", "value.is_empty()", 1);
    assert_ne!(source, mutated, "whitespace mutation did not apply");
    assert!(
        !mutated.contains("value.trim().is_empty()"),
        "mutation audit failed to remove whitespace rejection"
    );
}

#[test]
fn resolved_batch_fields_are_private_and_only_the_seal_can_construct_them() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    assert_move_only_private_type(
        &source,
        "pub(crate) struct ResolvedResidentProducerBatchMemoryV4 {",
        "ResolvedResidentProducerBatchMemoryV4",
    );
    for getter in [
        "pub(crate) const fn producer(&self)",
        "pub(crate) const fn first_column(&self)",
        "pub(crate) const fn column_count(&self)",
        "pub(crate) const fn additional_retained_bytes(&self)",
        "pub(crate) const fn scratch_bytes(&self)",
    ] {
        assert!(
            source.contains(getter),
            "missing read-only batch getter {getter:?}"
        );
    }
    assert_eq!(
        source
            .matches("producer_batches.push(ResolvedResidentProducerBatchMemoryV4 {")
            .count(),
        1,
        "resolved batches must be constructed exactly once inside the global seal"
    );
    for behavioral_test in [
        "valid_seven_producer_seal_assigns_monotonic_global_routes_and_batches",
        "producer_reordering_and_transform_pseudo_columns_fail_closed",
        "missing_producers_and_batch_gaps_or_overlaps_fail_closed",
    ] {
        assert!(
            source.contains(behavioral_test),
            "missing mutation fixture {behavioral_test}"
        );
    }
}

#[test]
fn mutation_audit_rejects_clone_derive_and_any_public_resolved_field() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    let cloned = source.replacen(
        "#[derive(Debug)]\npub(crate) struct ResidentRouteDraftV4 {",
        "#[derive(Debug, Clone)]\npub(crate) struct ResidentRouteDraftV4 {",
        1,
    );
    assert_ne!(source, cloned, "Clone mutation did not apply");
    assert!(
        std::panic::catch_unwind(|| {
            assert_move_only_private_type(
                &cloned,
                "pub(crate) struct ResidentRouteDraftV4 {",
                "ResidentRouteDraftV4",
            )
        })
        .is_err(),
        "source contract accepted Clone on a move-only route draft"
    );

    let public_field = source.replacen(
        "pub(crate) struct ResolvedResidentProducerBatchMemoryV4 {\n    producer: ResidentFeatureProducerV3,\n    first_column: usize,",
        "pub(crate) struct ResolvedResidentProducerBatchMemoryV4 {\n    producer: ResidentFeatureProducerV3,\n    pub first_column: usize,",
        1,
    );
    assert_ne!(source, public_field, "public-field mutation did not apply");
    assert!(
        std::panic::catch_unwind(|| {
            assert_move_only_private_type(
                &public_field,
                "pub(crate) struct ResolvedResidentProducerBatchMemoryV4 {",
                "ResolvedResidentProducerBatchMemoryV4",
            )
        })
        .is_err(),
        "source contract accepted a public resolved-memory field"
    );
}

#[test]
fn column_schema_seal_does_not_create_runtime_or_plan_authority() {
    let source = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    assert!(source.contains("pub(crate) struct SealedResidentColumnSchemaV4"));
    let column_only = format!(
        "{}{}",
        section(
            &source,
            "pub(crate) struct SealedResidentColumnSchemaV4",
            "impl SealedResidentColumnSchemaV4",
        ),
        section(
            &source,
            "impl ResidentColumnSchemaAssemblerV4",
            "fn derive_dataset_recipe_sha256_v4",
        )
    );
    for forbidden in [
        "GpuOnlyResidentAdmissionV3",
        "FeaturePlanV1",
        "DatasetFeatureArtifactProvenanceV1",
        "SourceArtifactBindingV1",
        "admission_identity_sha256",
        "final_feature_plan",
        "source_provenance",
    ] {
        assert!(
            !column_only.contains(forbidden),
            "column-only seal overclaimed authority through {forbidden:?}"
        );
    }
    assert!(
        source.contains("finalize_after_normalization_v4"),
        "final plan authority must exist only in the post-fit identity template"
    );
}
