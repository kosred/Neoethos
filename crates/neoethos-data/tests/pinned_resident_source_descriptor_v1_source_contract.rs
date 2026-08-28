use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-data"))
}

fn read(relative: &str) -> String {
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
        .unwrap_or_else(|| panic!("missing source-descriptor token {start:?}"))
        .1
        .split_once(end)
        .unwrap_or_else(|| panic!("missing source-descriptor terminator {end:?}"))
        .0
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
fn descriptor_is_move_only_crate_private_and_retains_artifacts() {
    let source = read("src/core/pinned_canonical_series_v1.rs");
    assert_move_only_private_type(
        &source,
        "pub(crate) struct PinnedResidentCanonicalSourceDescriptorV1 {",
        "PinnedResidentCanonicalSourceDescriptorV1",
    );
    let descriptor = section(
        &source,
        "pub(crate) struct PinnedResidentCanonicalSourceDescriptorV1 {",
        "\n}",
    );
    for required in ["receipt:", "sources:"] {
        assert!(
            descriptor.contains(required),
            "missing retained field {required:?}"
        );
    }
    for forbidden in [
        "pub fn into_resident_source_descriptor_v1",
        "from_hash",
        "from_bytes",
    ] {
        assert!(
            !source.contains(forbidden),
            "descriptor exposes {forbidden:?}"
        );
    }
}

#[test]
fn mutation_audit_rejects_clone_derive_and_any_public_descriptor_field() {
    let source = read("src/core/pinned_canonical_series_v1.rs");
    let cloned = source.replacen(
        "#[derive(Debug)]\npub(crate) struct PinnedResidentCanonicalSourceDescriptorV1 {",
        "#[derive(Debug, Clone)]\npub(crate) struct PinnedResidentCanonicalSourceDescriptorV1 {",
        1,
    );
    assert_ne!(source, cloned, "Clone mutation did not apply");
    assert!(
        std::panic::catch_unwind(|| {
            assert_move_only_private_type(
                &cloned,
                "pub(crate) struct PinnedResidentCanonicalSourceDescriptorV1 {",
                "PinnedResidentCanonicalSourceDescriptorV1",
            )
        })
        .is_err(),
        "source contract accepted Clone on the resident source descriptor"
    );

    let public_field = source.replacen(
        "pub(crate) struct PinnedResidentCanonicalSourceDescriptorV1 {\n    receipt: CanonicalDatasetSeriesReceiptV1,",
        "pub(crate) struct PinnedResidentCanonicalSourceDescriptorV1 {\n    pub receipt: CanonicalDatasetSeriesReceiptV1,",
        1,
    );
    assert_ne!(source, public_field, "public-field mutation did not apply");
    assert!(
        std::panic::catch_unwind(|| {
            assert_move_only_private_type(
                &public_field,
                "pub(crate) struct PinnedResidentCanonicalSourceDescriptorV1 {",
                "PinnedResidentCanonicalSourceDescriptorV1",
            )
        })
        .is_err(),
        "source contract accepted a public descriptor field"
    );
}

#[test]
fn consuming_conversion_uses_every_exact_generation_without_decode() {
    let source = read("src/core/pinned_canonical_series_v1.rs");
    let conversion = section(
        &source,
        "pub(crate) fn into_resident_source_descriptor_v1(",
        "\n    }\n",
    );
    let conversion = compact(conversion);
    for required in [
        "self",
        "letPinnedCanonicalSeriesV1",
        "receipt.direct_timeframes()",
        ".zip(generations)",
        "CanonicalDatasetArtifactV1::from_manifest",
        "artifact.source_binding(resident_source_node_id_v1(",
        "sources.len()==receipt.direct_timeframes().len()",
        "PinnedResidentCanonicalSourceDescriptorV1{receipt,sources}",
    ] {
        assert!(
            conversion.contains(required),
            "conversion is missing exact-generation step {required:?}"
        );
    }
    for forbidden in [
        "materialize_pinned_canonical_timeframe_v1",
        "materialize_pinned_canonical_series_v1",
        "SourceSegmentV1::new",
        "source_node_id:",
        "segments:",
        "row_window",
    ] {
        assert!(
            !conversion.contains(forbidden),
            "resident descriptor conversion may not use {forbidden:?}"
        );
    }
}

#[test]
fn artifact_factory_is_crate_private_and_binding_is_full_generation_only() {
    let artifact = read("src/core/canonical_ohlcv.rs");
    assert!(
        artifact.contains("pub(crate) fn from_manifest("),
        "pinned resident conversion needs the private manifest+lease factory"
    );
    let full_binding = section(&artifact, "pub fn source_binding(", "\n    }\n");
    for required in [
        "SourceSegmentV1::new(",
        "0,",
        "self.source_row_count",
        "self.source_timestamp_start_ms",
        "self.source_timestamp_end_ms",
    ] {
        assert!(
            full_binding.contains(required),
            "full-generation binding lost {required:?}"
        );
    }
}

#[test]
fn descriptor_exposes_read_only_binding_lookup_not_artifact_ownership() {
    let source = read("src/core/pinned_canonical_series_v1.rs");
    let source = compact(&source);
    for required in [
        "pub(crate)constfnreceipt(&self)",
        "pub(crate)fnsource_binding(&self,timeframe:CanonicalTimeframe",
        "pub(crate)fngeneration_count(&self)",
    ] {
        assert!(
            source.contains(required),
            "missing read-only getter {required:?}"
        );
    }
    for forbidden in [
        "pub(crate)fnartifact(",
        "pub(crate)fnlease(",
        "pub(crate)fninto_sources(",
        "pub(crate)fninto_artifacts(",
    ] {
        assert!(
            !source.contains(forbidden),
            "descriptor leaks retained owner via {forbidden:?}"
        );
    }
}

#[test]
fn descriptor_consumes_every_pinned_generation_into_ordered_resident_frames() {
    let source = read("src/core/pinned_canonical_series_v1.rs");
    let compact = compact(&source);
    for required in [
        "pub(crate)structMaterializedPinnedResidentCanonicalSourceV1",
        "pub(crate)structMaterializedPinnedResidentCanonicalSourcesV1",
        "pub(crate)fninto_materialized_resident_sources_v1(self,base_timeframe:CanonicalTimeframe",
        "receipt.direct_timeframes().iter().zip(sources)",
        "artifact.lease().reopen_verified()",
        "crate::vortex_array_to_ohlcv(array)",
        "CanonicalOhlcvFrame::from_parts(ohlcv,artifact)",
        "source_binding_sha256_v1(&binding)",
        "direct_parents.push(materialized)",
        "timeframe>base_timeframe",
        "base.is_none()",
    ] {
        assert!(
            compact.contains(required),
            "pinned resident materialization is missing {required:?}"
        );
    }
    for forbidden in [
        "row_window(",
        "load_canonical_timeframe(",
        "load_exact_canonical_timeframe(",
        "open_current_dataset_generation(",
    ] {
        let materialize = section(
            &source,
            "pub(crate) fn into_materialized_resident_sources_v1(",
            "\n    }\n",
        );
        assert!(
            !materialize.contains(forbidden),
            "pinned resident materialization may not use {forbidden:?}"
        );
    }
}

#[test]
fn materialized_source_owners_are_private_move_only_and_hash_full_bindings() {
    let source = read("src/core/pinned_canonical_series_v1.rs");
    for (declaration, type_name) in [
        (
            "pub(crate) struct MaterializedPinnedResidentCanonicalSourceV1 {",
            "MaterializedPinnedResidentCanonicalSourceV1",
        ),
        (
            "pub(crate) struct MaterializedPinnedResidentCanonicalSourcesV1 {",
            "MaterializedPinnedResidentCanonicalSourcesV1",
        ),
    ] {
        assert_move_only_private_type(&source, declaration, type_name);
    }
    let hash = section(&source, "fn source_binding_sha256_v1(", "\n}\n");
    for required in [
        "neoethos.data.full-source-artifact-binding.v1",
        "binding.source_node_id()",
        "binding.dataset_identity().canonical_bytes()",
        "binding.manifest_schema_id()",
        "binding.manifest_hash()",
        "binding.generation_id()",
        "binding.vortex_hash()",
        "binding.segments()",
    ] {
        assert!(hash.contains(required), "binding hash omits {required:?}");
    }
}
