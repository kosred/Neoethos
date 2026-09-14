use std::{fs, path::PathBuf};

fn crate_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join(path)).expect("read production source")
}

#[test]
fn resident_adaptive_view_checks_both_independent_caps_before_native_binding() {
    let resident = crate_file("src/resident_feature_store_v3.rs");
    let (_, tail) = resident
        .split_once("pub(crate) fn bind_evaluation_view_with_resident_adaptive_base_v1(")
        .expect("private adaptive binder");
    let (body, _) = tail.split_once("\n    }").expect("adaptive binder end");
    let has_both_pre_native_guards = |source: &str| {
        let Some(native) =
            source.find(".bind_evaluation_view_with_resident_adaptive_base_v1(view, request)")
        else {
            return false;
        };
        let checks = &source[..native];
        [
            "u64::try_from(view.ordered_index_values().map_or(0, <[u64]>::len))",
            "ordered_rows > limits.max_ordered_index_count()",
            "view_rows > limits.max_adaptive_row_count()",
            "request.parent_row_count() != parent_rows",
            "request.view_row_count() != view_rows",
            "return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(",
        ]
        .iter()
        .all(|check| checks.contains(check))
    };
    assert!(has_both_pre_native_guards(body));
    for guard in [
        "ordered_rows > limits.max_ordered_index_count()",
        "view_rows > limits.max_adaptive_row_count()",
    ] {
        let missing_guard = body.replacen(guard, "false", 1);
        assert_ne!(missing_guard, body);
        assert!(
            !has_both_pre_native_guards(&missing_guard),
            "neither independent workspace cap may stand in for the other"
        );
    }
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
    assert!(!native.contains("resident_adaptive_median_kernel_v1"));
    assert!(!native.contains("ResidentAdaptiveControlV1"));
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
    for kernel in [
        "resident_adaptive_parkinson_kernel_v1<<<",
        "resident_adaptive_rolling_sigma_kernel_v1<<<",
        "resident_adaptive_distance_kernel_v1<<<",
        "resident_adaptive_validate_normalized_kernel_v1<<<",
    ] {
        assert_eq!(bind.matches(kernel).count(), 1, "exactly one {kernel}");
    }
    assert!(bind.contains("kernel_submissions += 4ull"));
    assert!(
        bind.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .contains("? session->view_indices : nullptr")
    );
    assert!(bind.contains("view->adaptive_base_pips != nullptr"));
    assert!(bind.contains("view->adaptive_base_pips_len != 0"));
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
    assert!(!native.contains("resident_adaptive_control"));
    assert!(native.contains("if (!any_available)"));
    assert!(rust.contains("causal-unavailable-qnan=0x7ff8000000000000"));
    assert!(rust.contains("no-median-or-minimum-clamp"));
    assert!(
        rust.contains("normalized-arithmetic=whole-view-refusal-stricter-than-cpu-row-omission")
    );
}

#[test]
fn normalized_adaptive_base_rejects_tiny_pip_overflow_before_fixed_stop_fallback() {
    let native = crate_file("native/prototype_b_population.cu");

    assert!(
        native.contains("resident_adaptive_validate_normalized_kernel_v1"),
        "the final distance/pip normalization needs an all-row classified validation"
    );
    assert!(
        native.contains("classify_resident_adaptive_cell_v1(output[row])"),
        "normalization faults must be distinguished from canonical causal NaNs"
    );
    assert!(native.contains("cell == ResidentAdaptiveCellClassV1::ArithmeticFault"));
    assert!(native.contains("return isfinite(value) && value > 0.0"));
    assert!(native.contains("? distance / request.pip_size"));
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
fn resident_adaptive_warmup_is_causal_and_distance_is_computed_once() {
    let native = crate_file("native/prototype_b_population.cu");
    let (_, distance_tail) = native
        .split_once("__global__ void resident_adaptive_distance_kernel_v1(")
        .unwrap();
    let (distance, _) = distance_tail.split_once("\n}\n").unwrap();
    let causal = |body: &str| {
        let Some(warmup) = body.find("if (bar < kResidentAdaptiveTailWindowV1)") else {
            return false;
        };
        let Some(unavailable) = body.find("output[bar] = resident_adaptive_unavailable_v1();")
        else {
            return false;
        };
        let Some(tail) = body.find("resident_adaptive_tail_es_v1(dataset, bar, request.tail_step)")
        else {
            return false;
        };
        warmup < unavailable
            && unavailable < tail
            && body
                .matches("resident_adaptive_tail_es_v1(dataset, bar, request.tail_step)")
                .count()
                == 1
            && body.contains("isfinite(distance) && distance > 0.0")
            && body.contains(": resident_adaptive_unavailable_v1()")
            && !body.contains("median")
            && !body.contains("1.0e-9")
    };
    // The word median occurs only in an explanatory comment; inspect executable
    // lines so the negative control targets the former future-data fill.
    let executable = distance
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(causal(&executable));
    let missing = executable.replacen("if (bar < kResidentAdaptiveTailWindowV1)", "if (false)", 1);
    assert_ne!(missing, executable);
    assert!(!causal(&missing));
    assert!(!causal(&format!("{executable}\noutput[bar] = median;")));
    assert!(!native.contains("resident_adaptive_select_v1"));
}

#[test]
fn population_view_host_sources_are_owned_until_stream_retirement() {
    let native = crate_file("native/prototype_b_population.cu");
    let (_, bind_tail) = native
        .split_once("extern \"C\" std::int32_t neoethos_gpu_cuda_population_bind_view_v1(")
        .unwrap();
    let (bind, _) = bind_tail.split_once("\n}\n").unwrap();
    let bind = bind.split_whitespace().collect::<Vec<_>>().join(" ");
    for required in [
        "new (std::nothrow) ViewHostStagingV1{}",
        "std::memcpy(staging->ordered_indices, view->ordered_indices, static_cast<std::size_t>(ordered_upload_bytes))",
        "std::memcpy(staging->adaptive_base_pips, view->adaptive_base_pips, static_cast<std::size_t>(adaptive_upload_bytes))",
        "copy_to_device(session->view_indices, staging->ordered_indices",
        "copy_to_device(session->adaptive_base_pips, staging->adaptive_base_pips",
        "release_view_host_staging_v1",
        "return retire_staging(status)",
    ] {
        assert!(
            bind.contains(required),
            "missing owned transfer path: {required}"
        );
    }
    assert!(!bind.contains("copy_to_device(session->view_indices, view->ordered_indices"));
    assert!(!bind.contains("copy_to_device(session->adaptive_base_pips, view->adaptive_base_pips"));
    let (_, release_tail) = native
        .split_once("std::int32_t release_staging_after_stream_v1(")
        .unwrap();
    let (release, _) = release_tail.split_once("\n}\n").unwrap();
    let shared = crate_file("native/resident_host_staging_v1.hpp");
    let checked_retirement = |adapter: &str, owner: &str| {
        adapter.contains("neoethos_host_staging_v1::retire_after_stream_v1(")
            && adapter.contains("cudaLaunchHostFunc(stream, release, ticket) == cudaSuccess")
            && adapter.contains("cudaStreamSynchronize(stream) == cudaSuccess")
            && !adapter.contains("delete staging")
            && owner.contains("if (synchronize()) ticket->release_payload_once();")
            && owner.contains("delete payload_.exchange(nullptr, std::memory_order_acq_rel);")
            && owner.contains("ticket->release_submitter();")
    };
    assert!(checked_retirement(release, &shared));
    for (before, after) in [
        (
            "cudaLaunchHostFunc(stream, release, ticket)",
            "cudaLaunchHostFunc(stream, release, staging)",
        ),
        ("cudaStreamSynchronize(stream) == cudaSuccess", "true"),
    ] {
        assert!(
            release.contains(before),
            "negative control must alter actual adapter"
        );
        assert!(!checked_retirement(
            &release.replacen(before, after, 1),
            &shared
        ));
    }
    let premature_release = shared.replacen(
        "if (synchronize()) ticket->release_payload_once();",
        "ticket->release_payload_once();",
        1,
    );
    assert_ne!(premature_release, shared);
    assert!(!checked_retirement(release, &premature_release));
}

#[test]
fn failed_ordinary_view_bind_poison_precedes_any_reuse_of_previous_authority() {
    let rust = crate_file("src/population.rs");
    let (_, tail) = rust.split_once("pub fn bind_evaluation_view_v1(").unwrap();
    let (body, _) = tail.split_once("\n    }").unwrap();
    let closes_failed_bind = |source: &str| {
        let Some((_, native_tail)) =
            source.split_once("neoethos_gpu_cuda_population_bind_view_v1(self.handle, &raw)")
        else {
            return false;
        };
        let Some((_, failure_tail)) = native_tail.split_once("if status != STATUS_OK {") else {
            return false;
        };
        let Some((failure, _)) = failure_tail.split_once("\n        }") else {
            return false;
        };
        let Some(retained) = failure.find("self.bound_view_source_v1 = Some(view);") else {
            return false;
        };
        let Some(poisoned) =
            failure.find("self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;")
        else {
            return false;
        };
        let Some(returned) = failure.find("return Err(CudaPopulationError::native(") else {
            return false;
        };
        retained < returned && poisoned < returned
    };
    assert!(body.contains("self.require_strict_idle_v1(\"bind_evaluation_view_v1\")?"));
    assert!(closes_failed_bind(body));
    let missing_poison = body.replacen(
        "self.strict_resident_state = StrictResidentSessionStateV1::Poisoned;",
        "",
        1,
    );
    assert_ne!(missing_poison, body);
    assert!(!closes_failed_bind(&missing_poison));
    let missing_owner = body.replacen("self.bound_view_source_v1 = Some(view);", "", 1);
    assert_ne!(missing_owner, body);
    assert!(!closes_failed_bind(&missing_owner));
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
