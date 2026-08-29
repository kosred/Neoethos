const DISCOVERY: &str = include_str!("../src/app_services/discovery.rs");
const ENGINES_CONTROL: &str = include_str!("../src/server/engines_control.rs");

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, rest) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source marker {start:?}"));
    rest.split_once(end)
        .unwrap_or_else(|| panic!("missing source marker {end:?} after {start:?}"))
        .0
}

fn require_before(source: &str, earlier: &str, later: &str) {
    let earlier_at = source
        .find(earlier)
        .unwrap_or_else(|| panic!("missing required earlier boundary {earlier:?}"));
    let later_at = source
        .find(later)
        .unwrap_or_else(|| panic!("missing required later boundary {later:?}"));
    assert!(
        earlier_at < later_at,
        "{earlier:?} must execute before {later:?}"
    );
}

#[test]
fn app_acquires_exact_ordinal_strict_gpu_permit_before_pinning_or_materialization() {
    let handler = section(
        ENGINES_CONTROL,
        "pub async fn discovery_start(",
        "\n}\n\npub async fn discovery_stop",
    );

    for required in [
        "ExactCudaDeviceOrdinalV1",
        "StrictGpuOnlyFullDiscoveryPermitV1",
        "require_exact_cuda_device_ordinal_v1(",
        "require_all_full_discovery_stages_strict_gpu_v1(",
        "acquire_strict_gpu_only_full_discovery_permit_v1(",
        "PipelineStage::FULL_DISCOVERY",
    ] {
        assert!(
            handler.contains(required),
            "App Discovery admission is missing {required:?}"
        );
    }

    for later in ["pin_discovery_input(", "start_discovery_job("] {
        require_before(
            handler,
            "acquire_strict_gpu_only_full_discovery_permit_v1(",
            later,
        );
    }
    for forbidden in [
        "GPU_PREFERRED",
        "FallbackPolicy::AllowCpu",
        "DevicePreference::Auto",
        "StrictGpuOnlyFullDiscoveryPermitV1 {",
    ] {
        assert!(
            !handler.contains(forbidden),
            "App Discovery retains forbidden admission escape {forbidden:?}"
        );
    }
}

#[test]
fn app_request_carries_the_sealed_gpu_permit_into_the_worker() {
    let request = section(
        DISCOVERY,
        "pub struct DiscoveryRequest {",
        "\n}\n\nimpl DiscoveryRequest",
    );
    let worker = section(
        DISCOVERY,
        "pub fn start_discovery_job(",
        "\n/// On-disk contract between Discovery output",
    );

    assert!(
        request.contains("gpu_only_permit: StrictGpuOnlyFullDiscoveryPermitV1"),
        "DiscoveryRequest does not own the exact admission permit"
    );
    assert!(
        worker.contains("request.gpu_only_permit"),
        "the worker does not consume the request-bound GPU permit"
    );
    require_before(
        worker,
        "request.gpu_only_permit",
        "prepare_multitimeframe_features(",
    );
}

#[test]
fn app_worker_returns_only_a_sealed_compact_receipt_across_the_join_boundary() {
    let worker = section(
        DISCOVERY,
        "pub fn start_discovery_job(",
        "\n/// On-disk contract between Discovery output",
    );

    for required in [
        "run_discovery_cycle_gpu_only_compact_with_holdout_v1(",
        "SealedCompactGpuOnlyDiscoveryReceiptV1",
        "validate_against_gpu_only_permit(",
        "Ok::<SealedCompactGpuOnlyDiscoveryReceiptV1, anyhow::Error>(compact_receipt)",
        "completed_snapshot_from_compact_receipt_v1(",
        "write_model_targets_from_compact_receipt_v1(",
    ] {
        assert!(
            worker.contains(required),
            "App compact Search return boundary is missing {required:?}"
        );
    }

    let after_join = worker
        .split_once("let search_result = tokio::task::spawn_blocking")
        .expect("Search blocking boundary")
        .1
        .split_once(".await;")
        .expect("Search join boundary")
        .1;
    for forbidden in [
        "Ok(Ok(result)) => result",
        "completed_snapshot(base_snapshot, &result)",
        "write_model_targets_for_discovery(&request, &result)",
    ] {
        assert!(
            !after_join.contains(forbidden),
            "App returns a full Search payload across the join via {forbidden:?}"
        );
    }
}

#[test]
fn app_snapshot_boundary_cannot_consume_full_discovery_result() {
    assert!(
        DISCOVERY.contains(
            "pub fn completed_snapshot_from_compact_receipt_v1(\n    mut snapshot: JobSnapshot,\n    receipt: &SealedCompactGpuOnlyDiscoveryReceiptV1,"
        ),
        "completed App state is not built from the sealed compact receipt"
    );
    assert!(
        !DISCOVERY.contains(
            "pub fn completed_snapshot(mut snapshot: JobSnapshot, result: &DiscoveryResult)"
        ),
        "legacy full DiscoveryResult remains the App completion authority"
    );
}
