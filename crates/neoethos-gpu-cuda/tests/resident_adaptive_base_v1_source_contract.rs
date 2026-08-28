use std::{fs, path::PathBuf};

fn crate_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join(path)).expect("read production source")
}

#[test]
fn resident_adaptive_base_is_built_from_resident_prices_on_the_admitted_stream() {
    let header = crate_file("native/neoethos_gpu_cuda.h");
    let native = crate_file("native/prototype_b_population.cu");
    let rust = crate_file("src/population.rs");
    let resident = crate_file("src/resident_feature_store_v3.rs");

    assert!(header.contains("NeoResidentAdaptiveBaseRequestV1"));
    assert!(header.contains("neoethos_gpu_cuda_population_bind_resident_adaptive_view_v1"));
    assert!(rust.contains("pub struct ResidentAdaptiveBaseRequestV1"));
    assert!(rust.contains("struct ResidentAdaptiveBaseViewTokenV1"));
    assert!(resident.contains("pub(crate) fn bind_evaluation_view_with_resident_adaptive_base_v1"));
    assert!(
        resident.contains("pub fn bind_evaluation_view_with_resident_adaptive_base_checked_v1")
    );
    assert!(!resident.contains("pub fn bind_evaluation_view_with_resident_adaptive_base_v1("));
    assert!(!resident.contains("validate_current_adaptive_token_identity_v1"));
    assert!(rust.contains("arm_resident_adaptive_validator_guard_v1"));
    assert!(rust.contains("accept_resident_adaptive_validator_guard_v1"));
    assert!(rust.contains("poison_after_resident_adaptive_validator_rejection_v1"));

    assert!(native.contains("resident_adaptive_parkinson_kernel_v1"));
    assert!(native.contains("resident_adaptive_rolling_sigma_kernel_v1"));
    assert!(native.contains("resident_adaptive_distance_kernel_v1"));
    assert!(native.contains("resident_adaptive_median_kernel_v1"));
    assert!(native.contains("NEO_POPULATION_STATUS_ADAPTIVE_BASE_DEGENERATE"));
    assert!(native.contains("adaptive_upload_bytes must remain zero"));
    assert!(native.contains("view->view_kind == NEO_POPULATION_VIEW_ORDERED_INDICES"));

    let bind_start = native
        .find("neoethos_gpu_cuda_population_bind_resident_adaptive_view_v1(")
        .expect("native resident adaptive bind");
    let bind_tail = &native[bind_start..];
    let bind_end = bind_tail
        .find("\n}\n")
        .map(|offset| offset + 3)
        .expect("native resident adaptive bind end");
    let bind = &bind_tail[..bind_end];
    assert!(
        !bind.contains("copy_to_device("),
        "adaptive base must not cross H2D"
    );
    assert!(
        !bind.contains("cudaMemcpy"),
        "adaptive base must remain device-resident"
    );
    assert!(
        !bind.contains("cudaStreamSynchronize"),
        "adaptive producer must remain stream ordered"
    );
}

#[test]
fn resident_adaptive_identity_and_degenerate_failure_are_fail_closed() {
    let native = crate_file("native/prototype_b_population.cu");
    let rust = crate_file("src/population.rs");

    assert!(rust.contains("neoethos.population.resident-adaptive-base-request.v1"));
    assert!(rust.contains("neoethos.population.resident-adaptive-view-token.v1"));
    assert!(rust.contains("STATUS_ADAPTIVE_BASE_DEGENERATE"));
    assert!(native.contains("kAdaptiveBaseDegenerateSentinelV1"));
    assert!(native.contains("adaptive_base_failed_v1"));
    assert!(
        native.contains("resident_adaptive_control shares gap_flags only before gap generation")
    );
}

#[test]
fn normalized_adaptive_base_rejects_tiny_pip_overflow_before_fixed_stop_fallback() {
    let native = crate_file("native/prototype_b_population.cu");

    assert!(
        native.contains("resident_adaptive_validate_normalized_kernel_v1"),
        "the final distance/pip normalization needs an all-row finite validation"
    );
    assert!(
        native.contains("!isfinite(output[row])"),
        "a positive but tiny pip size can overflow a finite distance to infinity"
    );
    assert!(
        native.contains("output[0] = kAdaptiveBaseDegenerateSentinelV1"),
        "invalid normalized bases must poison resident evaluation instead of falling back to fixed stops"
    );

    let final_distance = native
        .rfind("resident_adaptive_distance_kernel_v1<<<")
        .expect("final resident adaptive distance launch");
    let validation = native
        .rfind("resident_adaptive_validate_normalized_kernel_v1<<<")
        .expect("normalized finite-validation launch");
    assert!(
        validation > final_distance,
        "finite validation must execute after final distance/pip normalization"
    );
}

#[test]
fn adaptive_and_quant_share_one_exact_cpu_cuda_log_schedule() {
    let adaptive = crate_file("native/prototype_b_population.cu");
    let quant = crate_file("native/resident_quant_v3.cu");
    let shared = crate_file("native/resident_exact_log_v3.cuh");
    let rust = crate_file("src/population.rs");
    let data_mod = crate_file("../neoethos-data/src/core/mod.rs");
    let data_lib = crate_file("../neoethos-data/src/lib.rs");

    assert!(adaptive.contains("#include \"resident_exact_log_v3.cuh\""));
    assert!(quant.contains("#include \"resident_exact_log_v3.cuh\""));
    assert!(shared.contains("exact_log_positive_f64_v3"));
    assert!(shared.contains("__dadd_rn"));
    assert!(shared.contains("__dsub_rn"));
    assert!(shared.contains("__dmul_rn"));
    assert!(shared.contains("__ddiv_rn"));
    assert!(
        adaptive.contains("exact_log_positive_f64_v3(fmax(value, 1.0e-12)"),
        "adaptive safe_log must use the frozen exact CUDA schedule"
    );
    assert!(
        !adaptive.contains("return log(fmax(value, 1.0e-12))"),
        "native libm log is not the zero-bit CPU authority"
    );
    assert!(rust.contains("cpu-cuda-bit-tolerance=zero"));
    assert!(!rust.contains("ulp-tolerance-not-bitwise"));
    assert!(data_mod.contains("pub mod quant_exact_math_v3;"));
    assert!(data_lib.contains("quant_log_positive_f64_v3"));
    assert!(data_lib.contains("QUANT_LOG_OPERATION_SCHEDULE_V3"));
}
