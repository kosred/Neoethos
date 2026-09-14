use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
}

fn read_required(relative: &str) -> String {
    let path = manifest_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"))
        .0
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(
            source.contains(token),
            "resident generation post-GA native bridge is missing {token:?}"
        );
    }
}

#[test]
fn native_bridge_reuses_the_sealed_generation_run_event_stream_and_allocation_in_place() {
    let cuda = read_required("native/resident_generation_v1.cu");
    let begin = section(
        &cuda,
        "extern \"C\" std::int32_t begin_resident_post_ga_in_place_v1(",
        "\n}\n\n",
    );
    require_all(
        begin,
        &[
            "run->sealed",
            "!run->post_ga_in_place_bound",
            "dependency->event_id == run->next_event_id",
            "dependency->generation_index == run->current_generation_index",
            "dependency->same_stream_enqueue_count == run->same_stream_enqueue_count",
            "consume_resident_generation_event_dependency_v1(run)",
            "run->post_ga_in_place_bound = true",
            "receipt->ready_event_id = dependency->event_id",
            "receipt->current_generation_index = run->current_generation_index",
            "receipt->same_stream_enqueue_count = run->same_stream_enqueue_count",
            "receipt->logical_population_count = run->logical_population_count",
            "receipt->retained_evaluation_capacity = run->retained_evaluation_capacity",
            "receipt->generation_allocation_total_device_bytes",
            "run->allocation.total_device_bytes;",
            "receipt->additional_allocation_count = 0",
            "receipt->additional_device_bytes = 0",
        ],
    );
    require_all(
        &cuda,
        &[
            "run->gene_scalars_device != nullptr",
            "run->gene_indices_device != nullptr",
            "run->gene_weights_device != nullptr",
            "run->metric_rows_device != nullptr",
            "run->resident_decision_keys_device != nullptr",
            "generation_content_identity_handle_v1(run->run_token, 1)",
            "generation_content_identity_handle_v1(run->run_token, 2)",
            "generation_content_identity_handle_v1(run->run_token, 3)",
        ],
    );
    for forbidden in [
        "new ",
        "delete ",
        "cudaMalloc",
        "cudaFree",
        "cudaEventCreate",
        "cudaEventDestroy",
        "cudaStreamCreate",
        "cudaSetDevice",
        "cudaMemcpy",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
    ] {
        assert!(
            !begin.contains(forbidden),
            "in-place native bridge creates, transfers, synchronizes, or frees via {forbidden:?}"
        );
    }
}
