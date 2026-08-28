use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalPinnedSourceBindingFactsV1,
    CanonicalPinnedSourceProjectionV1, CanonicalPinnedSourceSegmentFactsV1, CanonicalTimeframe,
};

fn binding(
    identity: CanonicalDatasetIdentity,
    manifest_byte: u8,
    vortex_byte: u8,
    row_end: u64,
) -> CanonicalPinnedSourceBindingFactsV1 {
    CanonicalPinnedSourceBindingFactsV1::checked_new(
        identity,
        "neoethos.canonical-dataset-manifest.v1",
        [manifest_byte; 32],
        format!("g1-{}", "a".repeat(64)),
        [vortex_byte; 32],
        BarTimestampConvention::BarOpen,
        vec![
            CanonicalPinnedSourceSegmentFactsV1::checked_new(
                0,
                row_end,
                1_700_000_000_000,
                1_700_000_060_000,
            )
            .expect("segment"),
        ],
    )
    .expect("binding")
}

#[test]
fn projection_is_node_name_independent_but_binds_every_exact_source_fact() {
    let m1 = CanonicalDatasetIdentity::external(
        "projection-contract",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("M1 identity");
    let h1 = CanonicalDatasetIdentity::external(
        "projection-contract",
        "EURUSD",
        CanonicalTimeframe::H1,
        BarTimestampConvention::BarOpen,
    )
    .expect("H1 identity");

    let cpu_bindings = vec![binding(m1.clone(), 1, 2, 2), binding(h1.clone(), 3, 4, 2)];
    let native_bindings = vec![binding(h1, 3, 4, 2), binding(m1.clone(), 1, 2, 2)];
    let cpu = CanonicalPinnedSourceProjectionV1::checked_from_binding_facts_v1(
        m1.clone(),
        2,
        cpu_bindings,
    )
    .expect("CPU projection");
    let native = CanonicalPinnedSourceProjectionV1::checked_from_binding_facts_v1(
        m1.clone(),
        2,
        native_bindings,
    )
    .expect("native projection");

    assert_eq!(
        cpu, native,
        "source node vocabulary must not enter identity"
    );
    assert_eq!(cpu.anchor_dataset_identity(), &m1);
    assert_eq!(cpu.base_timeframe(), CanonicalTimeframe::M1);
    assert_eq!(cpu.parent_row_count(), 2);
    assert_eq!(cpu.bindings().len(), 2);
    assert_ne!(cpu.identity_sha256(), [0; 32]);

    let changed_segment = vec![
        binding(m1.clone(), 1, 2, 2),
        binding(native.bindings()[1].dataset_identity().clone(), 3, 4, 3),
    ];
    let changed =
        CanonicalPinnedSourceProjectionV1::checked_from_binding_facts_v1(m1, 2, changed_segment)
            .expect("changed projection");
    assert_ne!(cpu, changed, "exact consumed segments must enter identity");
}

#[test]
fn prepared_and_materialized_carriers_expose_the_same_typed_projection() {
    let source = std::fs::read_to_string(format!(
        "{}/src/core/gpu_resident_feature_store_v3.rs",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("read production Data bridge");

    assert!(source.contains("pinned_source_projection_v1: CanonicalPinnedSourceProjectionV1"));
    assert!(source.contains("pub const fn pinned_source_projection_v1("));
    assert!(source.contains("derive_pinned_source_projection_v1(preflight.resident_sources())"));
    assert!(source.contains("prepared pinned-source projection does not match"));
}
