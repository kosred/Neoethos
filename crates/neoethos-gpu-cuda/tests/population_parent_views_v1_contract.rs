use neoethos_gpu_cuda::{
    CudaPopulationDeviceIdentityV1, PopulationEvaluationViewV1, PopulationParentDatasetInputV1,
    PopulationParentDatasetV1, PopulationResidencyCountersV1, PopulationTimestampModeV1,
    PopulationViewKindV1, cuda_build_manifest_v1,
};
use std::sync::Arc;

const SMC_SLOTS: usize = 11;
const HEADER: &str = include_str!("../native/neoethos_gpu_cuda.h");
const CUDA: &str = include_str!("../native/prototype_b_population.cu");
const STUB: &str = include_str!("../native/stub.cpp");
const RUST: &str = include_str!("../src/population.rs");

fn shared<T>(values: Vec<T>) -> Arc<[T]> {
    Arc::from(values)
}

fn parent(rows: usize, features: usize) -> PopulationParentDatasetV1 {
    PopulationParentDatasetV1::new(PopulationParentDatasetInputV1 {
        close: shared((0..rows).map(|row| 1.0 + row as f64).collect()),
        high: shared((0..rows).map(|row| 1.5 + row as f64).collect()),
        low: shared((0..rows).map(|row| 0.5 + row as f64).collect()),
        indicators_feature_major: shared(
            (0..features * rows)
                .map(|offset| offset as f64 / 10.0)
                .collect(),
        ),
        feature_count: features,
        months: shared(vec![1; rows]),
        days: shared(vec![2; rows]),
        timestamps: shared(
            (0..rows)
                .map(|row| 1_700_000_000_000 + row as i64)
                .collect(),
        ),
        smc_rows: shared(vec![0; rows * SMC_SLOTS]),
    })
    .expect("exact immutable parent")
}

fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing source boundary `{start}`"));
    let tail = &source[start..];
    let end = tail
        .find(end)
        .unwrap_or_else(|| panic!("missing source boundary `{end}` after `{start}`"));
    &tail[..end]
}

#[test]
fn parent_dataset_is_immutable_exact_and_excludes_view_local_adaptive_state() {
    let parent = parent(8, 3);

    assert_eq!(parent.row_count(), 8);
    assert_eq!(parent.feature_count(), 3);
    assert_eq!(parent.indicators_feature_major().len(), 24);
    assert!(RUST.contains("pub struct PopulationParentDatasetV1"));
    let parent_source = source_between(
        RUST,
        "pub struct PopulationParentDatasetV1",
        "pub enum PopulationViewKindV1",
    );
    assert!(!parent_source.contains("adaptive_base_pips"));
}

#[test]
fn parent_shape_or_non_finite_value_is_refused_before_ffi() {
    let wrong_shape = PopulationParentDatasetV1::new(PopulationParentDatasetInputV1 {
        close: shared(vec![1.0; 4]),
        high: shared(vec![1.0; 3]),
        low: shared(vec![1.0; 4]),
        indicators_feature_major: shared(vec![1.0; 8]),
        feature_count: 2,
        months: shared(vec![1; 4]),
        days: shared(vec![1; 4]),
        timestamps: shared(vec![1; 4]),
        smc_rows: shared(vec![0; 4 * SMC_SLOTS]),
    })
    .unwrap_err();
    assert!(wrong_shape.to_string().contains("high"));

    let non_finite = PopulationParentDatasetV1::new(PopulationParentDatasetInputV1 {
        close: shared(vec![1.0, f64::NAN]),
        high: shared(vec![1.0; 2]),
        low: shared(vec![1.0; 2]),
        indicators_feature_major: shared(vec![1.0; 4]),
        feature_count: 2,
        months: shared(vec![1; 2]),
        days: shared(vec![1; 2]),
        timestamps: shared(vec![1; 2]),
        smc_rows: shared(vec![0; 2 * SMC_SLOTS]),
    })
    .unwrap_err();
    assert!(non_finite.to_string().contains("close[1]"));
}

#[test]
fn full_and_contiguous_views_are_scalar_descriptors_without_index_uploads() {
    let full =
        PopulationEvaluationViewV1::full(8, PopulationTimestampModeV1::Canonical, None).unwrap();
    assert_eq!(full.kind(), PopulationViewKindV1::Full);
    assert_eq!(full.row_count(), 8);
    assert_eq!(full.ordered_index_values(), None);

    let range = PopulationEvaluationViewV1::contiguous_range(
        8,
        2,
        7,
        PopulationTimestampModeV1::Canonical,
        None,
    )
    .unwrap();
    assert_eq!(range.kind(), PopulationViewKindV1::ContiguousRange);
    assert_eq!(range.range(), Some(2..7));
    assert_eq!(range.row_count(), 5);
    assert_eq!(range.ordered_index_values(), None);
}

#[test]
fn ordered_view_is_a_compact_strictly_increasing_u64_device_map() {
    let ordered = PopulationEvaluationViewV1::ordered_indices(
        10,
        shared(vec![0_u64, 3, 9]),
        PopulationTimestampModeV1::DisabledIndexDelta,
        None,
    )
    .unwrap();
    assert_eq!(ordered.kind(), PopulationViewKindV1::OrderedIndices);
    assert_eq!(ordered.ordered_index_values(), Some(&[0_u64, 3, 9][..]));
    assert_eq!(ordered.row_count(), 3);

    for indices in [vec![], vec![0, 0], vec![3, 2], vec![0, 10]] {
        assert!(
            PopulationEvaluationViewV1::ordered_indices(
                10,
                shared(indices),
                PopulationTimestampModeV1::DisabledIndexDelta,
                None,
            )
            .is_err()
        );
    }
}

#[test]
fn adaptive_series_is_bound_and_uploaded_per_exact_view_only() {
    let adaptive = shared(vec![2.0; 5]);
    let range = PopulationEvaluationViewV1::contiguous_range(
        8,
        2,
        7,
        PopulationTimestampModeV1::Canonical,
        Some(adaptive.clone()),
    )
    .unwrap();
    assert_eq!(range.adaptive_base_pips(), Some(adaptive.as_ref()));

    let error = PopulationEvaluationViewV1::contiguous_range(
        8,
        2,
        7,
        PopulationTimestampModeV1::Canonical,
        Some(shared(vec![2.0; 8])),
    )
    .unwrap_err();
    assert!(error.to_string().contains("adaptive"));
}

#[test]
fn timestamp_mode_is_explicit_and_disabled_index_delta_never_reads_parent_deltas() {
    let canonical =
        PopulationEvaluationViewV1::full(4, PopulationTimestampModeV1::Canonical, None).unwrap();
    let disabled = PopulationEvaluationViewV1::ordered_indices(
        4,
        shared(vec![0, 2]),
        PopulationTimestampModeV1::DisabledIndexDelta,
        None,
    )
    .unwrap();

    assert_eq!(
        canonical.timestamp_mode(),
        PopulationTimestampModeV1::Canonical
    );
    assert_eq!(
        disabled.timestamp_mode(),
        PopulationTimestampModeV1::DisabledIndexDelta
    );
    assert!(CUDA.contains("population_timestamp_at"));
    assert!(CUDA.contains("NEO_POPULATION_TIMESTAMP_DISABLED_INDEX_DELTA"));
}

#[test]
fn native_abi_has_separate_parent_bind_and_counter_functions_in_cuda_and_stub() {
    for required in [
        "struct NeoPopulationParentDatasetV1",
        "struct NeoPopulationEvaluationViewV1",
        "struct NeoPopulationResidencyCountersV1",
    ] {
        assert!(HEADER.contains(required), "header is missing `{required}`");
    }
    for required in [
        "neoethos_gpu_cuda_population_upload_parent_v1",
        "neoethos_gpu_cuda_population_bind_view_v1",
        "neoethos_gpu_cuda_population_read_residency_counters_v1",
    ] {
        assert!(HEADER.contains(required), "header is missing `{required}`");
        assert!(STUB.contains(required), "stub is missing `{required}`");
    }
}

#[test]
fn parent_and_view_hot_paths_have_no_intermediate_stream_synchronization() {
    let parent_upload = source_between(
        CUDA,
        "neoethos_gpu_cuda_population_upload_parent_v1(",
        "neoethos_gpu_cuda_population_bind_view_v1(",
    );
    let bind = source_between(
        CUDA,
        "neoethos_gpu_cuda_population_bind_view_v1(",
        "neoethos_gpu_cuda_population_upload_genes(",
    );

    for source in [parent_upload, bind] {
        assert!(!source.contains("cudaStreamSynchronize"));
        assert!(!source.contains("cudaDeviceSynchronize"));
    }
    assert!(CUDA.contains("population_parent_row"));
    assert!(CUDA.contains("view_indices"));
}

#[test]
fn resident_evaluation_has_no_environment_controlled_synchronization_path() {
    let evaluate = source_between(
        CUDA,
        "neoethos_gpu_cuda_population_b_evaluate(",
        "neoethos_gpu_cuda_population_wait(",
    );
    assert!(!evaluate.contains("cudaStreamSynchronize"));
    assert!(!evaluate.contains("cudaDeviceSynchronize"));
    assert!(!CUDA.contains("NEOETHOS_GPU_STAGE_TIMING"));
    assert!(!CUDA.contains("std::getenv"));
}

#[test]
fn exact_residency_counters_distinguish_uploads_and_every_readback_class() {
    fn assert_counter_type<T: Copy + Default + Eq>() {}
    assert_counter_type::<PopulationResidencyCountersV1>();

    for getter in [
        "parent_upload_count",
        "parent_upload_bytes",
        "view_binding_count",
        "full_binding_count",
        "range_binding_count",
        "ordered_binding_count",
        "ordered_index_upload_bytes",
        "adaptive_upload_bytes",
        "stream_creation_count",
        "explicit_synchronization_count",
        "metric_rows_readback_count",
        "metric_rows_readback_rows",
        "metric_rows_readback_bytes",
        "diagnostic_readback_count",
        "diagnostic_readback_rows",
        "diagnostic_readback_bytes",
        "accepted_trade_total_readback_count",
        "accepted_trade_total_readback_bytes",
    ] {
        assert!(RUST.contains(getter), "missing exact counter `{getter}`");
    }

    assert!(!RUST.contains(concat!("compact_result_", "readback_count")));
    assert!(!CUDA.contains(concat!("compact_result_", "readback_count")));

    let wait = source_between(
        CUDA,
        "neoethos_gpu_cuda_population_wait(",
        "neoethos_gpu_cuda_population_read_metrics(",
    );
    assert!(wait.contains("accepted_trade_total_readback_count"));
    assert!(wait.contains("accepted_trade_total_readback_bytes"));

    let metrics = source_between(
        CUDA,
        "neoethos_gpu_cuda_population_read_metrics(",
        "neoethos_gpu_cuda_population_read_diagnostics(",
    );
    for required in [
        "metric_rows_readback_count",
        "metric_rows_readback_rows",
        "metric_rows_readback_bytes",
    ] {
        assert!(metrics.contains(required), "metric D2H omits `{required}`");
    }

    let diagnostics = &CUDA[CUDA
        .find("neoethos_gpu_cuda_population_read_diagnostics(")
        .expect("diagnostic readback boundary")..];
    for required in [
        "diagnostic_readback_count",
        "diagnostic_readback_rows",
        "diagnostic_readback_bytes",
    ] {
        assert!(
            diagnostics.contains(required),
            "diagnostic D2H omits `{required}`"
        );
    }
}

#[test]
fn native_session_exposes_exact_selected_device_and_build_identity() {
    fn assert_device_type<T: Copy + Default + Eq>() {}
    assert_device_type::<CudaPopulationDeviceIdentityV1>();
    let _: fn() -> Option<&'static str> = cuda_build_manifest_v1;
    assert!(RUST.contains("impl Default for CudaPopulationDeviceIdentityV1"));

    assert!(
        HEADER.contains("struct NeoPopulationDeviceIdentityV1"),
        "header is missing `struct NeoPopulationDeviceIdentityV1`"
    );
    let required = "neoethos_gpu_cuda_population_read_device_identity_v1";
    assert!(HEADER.contains(required), "header is missing `{required}`");
    assert!(STUB.contains(required), "stub is missing `{required}`");
    for required in [
        "selected_device_ordinal",
        "compute_capability_major",
        "compute_capability_minor",
        "multiprocessor_count",
        "total_global_memory_bytes",
        "pci_domain_id",
        "pci_bus_id",
        "pci_device_id",
        "uuid",
        "name",
    ] {
        assert!(
            RUST.contains(required),
            "device identity omits `{required}`"
        );
    }
}

#[test]
fn legacy_dataset_upload_is_compatibility_only_and_not_the_parent_resident_route() {
    let legacy = source_between(
        RUST,
        "pub fn upload_dataset(",
        "pub fn upload_parent_dataset_v1(",
    );
    assert!(legacy.contains("compatibility-only"));
    assert!(!legacy.contains("PopulationParentDatasetV1::new_unchecked"));
    assert!(!RUST.contains("from_hash"));
}
