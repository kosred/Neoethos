const ADAPTER: &str = include_str!("../src/gpu_native/prototype_b_population_eval.rs");
const DISCOVERY: &str = include_str!("../src/discovery.rs");
const EVAL: &str = include_str!("../src/eval.rs");
const ROUTE: &str = include_str!("../src/strict_discovery_device_route_v1.rs");
const RESIDENT_PARENT: &str =
    include_str!("../src/population_execution_evidence_v1/native_cuda_resident_v1.rs");

#[test]
fn strict_adapter_sizes_from_the_native_metrics_only_plan_not_outcomes() {
    assert!(
        ADAPTER.contains("PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1"),
        "the adapter must consume gpu-cuda's authoritative strict metrics-only byte plan"
    );
    assert!(
        !ADAPTER.contains("MAX_TRADES_PER_CANDIDATE * outcome"),
        "the strict adapter still charges 8,192 compatibility outcomes per scenario"
    );
}

#[test]
fn runtime_sizing_is_bound_to_one_admitted_snapshot_and_both_row_extents() {
    for required in [
        "pre_parent_free_memory_bytes",
        "resident_parent_rows",
        "evaluation_rows",
        "gene_count",
        "gene_term_count",
    ] {
        assert!(
            ADAPTER.contains(required) || ROUTE.contains(required),
            "missing strict run-bound sizing fact: {required}"
        );
    }
    assert!(
        !ADAPTER.contains("device_free_memory_bytes("),
        "runtime rebatching must reuse the admitted pre-parent snapshot, not query changing free VRAM"
    );
}

#[test]
fn live_parent_upload_is_typed_as_unsplittable_before_recursive_rebatching() {
    let upload = RESIDENT_PARENT
        .find(".upload_parent_dataset_v1(")
        .expect("live resident parent upload call");
    let after_upload = &RESIDENT_PARENT[upload..];
    let install = after_upload
        .find("*session = Some(")
        .expect("resident session installation after parent upload");
    let upload_path = &after_upload[..install];

    assert!(
        upload_path.contains("UnsplittablePopulationAllocationV1"),
        "the live immutable-parent allocation is still retryable by splitting an unrelated scenario list"
    );
    assert!(
        ADAPTER.contains("downcast_ref::<UnsplittablePopulationAllocationV1>()"),
        "the recursive retry classifier must recognize the live parent-allocation marker"
    );
}

#[test]
fn pre_run_population_auto_remains_fail_closed_until_a_persisted_receipt_exists() {
    let adapter_only = EVAL
        .find("#[cfg(all(feature = \"gpu-b-adapter\", not(feature = \"gpu\")))]")
        .expect("standalone Prototype-B pre-run sizing function");
    let body = &EVAL[adapter_only..];
    let end = body
        .find("/// Non-GPU build")
        .expect("next sizing variant must delimit adapter-only function");
    let body = &body[..end];
    assert!(body.contains("None"));
    assert!(!body.contains("device_free_memory_bytes"));
    assert!(
        !DISCOVERY.contains("resolve_population_auto_from_admitted_ceiling_v1"),
        "S3a must not partially resolve population_auto without a persisted sizing receipt"
    );
}
