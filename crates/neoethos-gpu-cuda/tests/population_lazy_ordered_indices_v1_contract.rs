use std::fs;
use std::path::PathBuf;

use neoethos_gpu_cuda::{
    PopulationGeneStorePlanV1, PopulationMetricsOnlyPlanV1, PopulationParentDevicePlanV1,
};

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

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing function signature {signature:?}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing function body for {signature:?}"));
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated function body for {signature:?}")
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
    let upload = function_body(&cuda, "neoethos_gpu_cuda_population_upload_parent_v1(");
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
    let bind = function_body(&cuda, "neoethos_gpu_cuda_population_bind_view_v1(");
    require_all(
        bind,
        &[
            "NEO_POPULATION_VIEW_FULL",
            "NEO_POPULATION_VIEW_CONTIGUOUS_RANGE",
            "NEO_POPULATION_VIEW_ORDERED_INDICES",
            "ensure_device_capacity_v3(&session->view_indices",
        ],
    );
    assert!(
        bind.find("NEO_POPULATION_VIEW_ORDERED_INDICES").unwrap()
            < bind
                .find("ensure_device_capacity_v3(&session->view_indices")
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
            "required <= *capacity",
            "ensure_device_capacity_v3(&session->view_indices",
            "&session->view_indices_capacity",
            "view_indices_capacity",
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
    let bind = function_body(&cuda, "neoethos_gpu_cuda_population_bind_view_v1(");
    require_all(
        &cuda,
        &[
            "std::size_t adaptive_base_pips_capacity = 0;",
            "ensure_device_capacity_v3(&session->adaptive_base_pips",
            "&session->adaptive_base_pips_capacity",
            "adaptive_base_pips_capacity",
        ],
    );
    let presence = bind
        .find("view->adaptive_base_pips != nullptr")
        .expect("adaptive presence guard");
    let growth = bind
        .find("ensure_device_capacity_v3(&session->adaptive_base_pips")
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
    require_all(
        &adapter,
        &[
            "checked_from_parent_and_view_extents_v1",
            "ordered_index_capacity",
            "adaptive_row_capacity",
            "evidence.ordered_index_capacity_v1()",
            "evidence.adaptive_row_capacity_v1()",
            "checked_add",
        ],
    );

    let immutable = PopulationParentDevicePlanV1::checked_from_parent_extents_v1(10_000, 257)
        .expect("checked immutable parent");
    let exact = PopulationParentDevicePlanV1::checked_from_parent_and_view_extents_v1(
        10_000, 257, 2_500, 4_000,
    )
    .expect("checked exact active view capacities");
    assert_eq!(exact.copied_parent_bytes(), immutable.copied_parent_bytes());
    assert_eq!(exact.gap_flags_bytes(), immutable.gap_flags_bytes());
    assert_eq!(exact.view_indices_bytes(), 2_500 * 8);
    assert_eq!(exact.adaptive_base_pips_bytes(), 4_000 * 8);
    assert_eq!(
        exact.total_device_bytes(),
        immutable.total_device_bytes() + (2_500 + 4_000) * 8
    );
}

#[test]
fn worked_large_parent_budget_refuses_twelve_gib_but_fits_sixteen_gib() {
    const GIB: u64 = 1024 * 1024 * 1024;
    const PARENT_ROWS: usize = 5_270_000;
    const STAGE1_ROWS: usize = 1_317_500;
    const FEATURES: usize = 257;

    let parent = PopulationParentDevicePlanV1::checked_from_parent_and_view_extents_v1(
        PARENT_ROWS,
        FEATURES,
        0,
        STAGE1_ROWS,
    )
    .expect("worked M1 parent and adaptive Stage-1 view");
    let genes = PopulationGeneStorePlanV1::checked_from_gene_extents_v1(200, 3_200)
        .expect("worked 200-gene store");
    let scenarios = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(16_384, 240)
        .expect("worked strict scenario workspace");
    let required = parent
        .total_device_bytes()
        .checked_add(genes.total_device_bytes())
        .and_then(|bytes| bytes.checked_add(scenarios.total_device_bytes()))
        .and_then(|bytes| bytes.checked_add(64 * 1024 * 1024))
        .expect("worked budget fits u64");
    let twelve_gib_budget = (12 * GIB / 10) * 7;
    let sixteen_gib_budget = (16 * GIB / 10) * 7;
    assert!(required > twelve_gib_budget);
    assert!(required <= sixteen_gib_budget);
}
