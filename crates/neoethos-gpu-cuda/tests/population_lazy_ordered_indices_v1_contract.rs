use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
}

fn read(relative: &str) -> String {
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
            "lazy native view is missing {token:?}"
        );
    }
}

#[test]
fn immutable_parent_upload_does_not_allocate_optional_view_buffers() {
    let cuda = read("native/prototype_b_population.cu");
    let upload = section(
        &cuda,
        "neoethos_gpu_cuda_population_upload_parent_v1(",
        "\n}\n\nextern \"C\" std::int32_t neoethos_gpu_cuda_population_bind_view_v1(",
    );
    for forbidden in [
        "device_alloc(&session->view_indices",
        "device_alloc(&session->adaptive_base_pips",
    ] {
        assert!(
            !upload.contains(forbidden),
            "parent upload reserves optional per-view memory via {forbidden:?}"
        );
    }
}

#[test]
fn full_and_range_views_need_no_device_index_map() {
    let cuda = read("native/prototype_b_population.cu");
    let bind = section(
        &cuda,
        "neoethos_gpu_cuda_population_bind_view_v1(",
        "\n}\n\nextern \"C\" std::int32_t neoethos_gpu_cuda_population_read_residency_counters_v1(",
    );
    require_all(
        bind,
        &[
            "NEO_POPULATION_VIEW_FULL",
            "NEO_POPULATION_VIEW_CONTIGUOUS_RANGE",
            "NEO_POPULATION_VIEW_ORDERED_INDICES",
            "ordered_index_required_capacity",
        ],
    );
    assert!(
        bind.find("NEO_POPULATION_VIEW_ORDERED_INDICES").unwrap()
            < bind
                .find("grow_device_buffer(&session->view_indices")
                .unwrap(),
        "index-map allocation must be guarded by the ordered-index view"
    );
}

#[test]
fn ordered_index_map_grows_to_required_capacity_and_reuses_larger_storage() {
    let cuda = read("native/prototype_b_population.cu");
    require_all(
        &cuda,
        &[
            "std::size_t view_indices_capacity = 0;",
            "rows > session->view_indices_capacity",
            "grow_device_buffer(&session->view_indices",
            "&session->view_indices_capacity",
            "ordered_index_capacity_bytes",
        ],
    );
    assert!(
        !cuda.contains("device_alloc(&session->view_indices, parent_rows)"),
        "ordered-index storage is still parent-sized"
    );
}

#[test]
fn adaptive_base_buffer_is_lazy_and_grows_only_for_a_present_view_series() {
    let cuda = read("native/prototype_b_population.cu");
    let bind = section(
        &cuda,
        "neoethos_gpu_cuda_population_bind_view_v1(",
        "\n}\n\nextern \"C\" std::int32_t neoethos_gpu_cuda_population_read_residency_counters_v1(",
    );
    require_all(
        &cuda,
        &[
            "std::size_t adaptive_base_pips_capacity = 0;",
            "rows > session->adaptive_base_pips_capacity",
            "&session->adaptive_base_pips_capacity",
            "adaptive_capacity_bytes",
        ],
    );
    let presence = bind
        .find("view->adaptive_base_pips != nullptr")
        .expect("adaptive presence guard");
    let growth = bind
        .find("grow_device_buffer(&session->adaptive_base_pips")
        .expect("adaptive lazy growth");
    assert!(
        presence < growth,
        "adaptive allocation is not presence-guarded"
    );
    assert!(
        !cuda.contains("device_alloc(&session->adaptive_base_pips, parent_rows)"),
        "optional adaptive storage is still parent-sized"
    );
}

#[test]
fn host_budget_charges_one_parent_matrix_plus_exact_active_view_capacities() {
    let adapter = read("../neoethos-search/src/gpu_native/prototype_b_population_eval.rs");
    let dataset = section(
        &adapter,
        "fn prototype_b_dataset_peak_bytes(",
        "\n}\n\nfn candidates_for_free_memory(",
    );
    assert_eq!(
        dataset.matches("indicator_bytes").count(),
        2,
        "one declaration and one addition must charge exactly one resident indicator matrix"
    );
    for stale in [
        "charged twice",
        "21.6 GB at the transpose peak",
        "the CPU lane is the honest answer",
    ] {
        assert!(
            !adapter.contains(stale),
            "VRAM authority retains stale claim {stale:?}"
        );
    }
    require_all(
        &adapter,
        &[
            "ordered_index_capacity_bytes",
            "adaptive_capacity_bytes",
            "checked_add",
            "StrictGpuMinimumBatchNotResident",
        ],
    );
}

#[test]
fn worked_large_parent_budget_refuses_twelve_gib_but_fits_sixteen_gib() {
    let adapter = read("../neoethos-search/src/gpu_native/prototype_b_population_eval.rs");
    require_all(
        &adapter,
        &[
            "12 * 1024 * 1024 * 1024",
            "16 * 1024 * 1024 * 1024",
            "5_270_000",
            "257",
            "Sizing::NoRoom",
            "Sizing::Fits",
        ],
    );
    assert!(
        !adapter.contains("if fits < 16 {\n        return Sizing::NoRoom"),
        "a small positive native fit still routes toward CPU instead of strict same-card batches"
    );
}
