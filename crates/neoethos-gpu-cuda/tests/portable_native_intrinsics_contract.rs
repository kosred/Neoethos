//! Host-only source regressions. These do not prove CUDA/HIP device execution.

const FIRST_HIT: &str = include_str!("../native/prototype_b.cu");
const GENERATION: &str = include_str!("../native/resident_generation_v1.cu");
const POPULATION: &str = include_str!("../native/prototype_b_population.cu");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn logical_subgroup_contract(source: &str) -> bool {
    let source = compact(source);
    [
        "constexprunsignedFIRST_HIT_LOGICAL_WIDTH_V1=32u;",
        "threadIdx.x&(FIRST_HIT_LOGICAL_WIDTH_V1-1u)",
        "bar+=FIRST_HIT_LOGICAL_WIDTH_V1",
        "constexprunsignedlonglongmask=0xffffffffull;",
        "intoffset=FIRST_HIT_LOGICAL_WIDTH_V1/2;offset>0;offset>>=1",
        "__shfl_down_sync(mask,best_bar,offset,FIRST_HIT_LOGICAL_WIDTH_V1)",
        "__shfl_down_sync(mask,best_reason,offset,FIRST_HIT_LOGICAL_WIDTH_V1)",
        "warp_first_hit_kernel<<<static_cast<unsigned>(event_count),FIRST_HIT_LOGICAL_WIDTH_V1>>>",
    ]
    .into_iter()
    .all(|required| source.contains(required))
        && source.matches("__shfl_down_sync(").count() == 2
}

fn trap_contract(source: &str) -> bool {
    let source = compact(source);
    let Some((_, helper)) = source.split_once("voidresident_generation_trap_v1(){") else {
        return false;
    };
    let Some((helper, _)) = helper.split_once('}') else {
        return false;
    };
    trap_body_contract(&source, helper)
}

fn trap_body_contract(source: &str, helper: &str) -> bool {
    let Some((_, amd_branch)) = helper.split_once("#ifdefined(__HIP_PLATFORM_AMD__)") else {
        return false;
    };
    let Some((amd_branch, cuda_branch)) = amd_branch.split_once("#else") else {
        return false;
    };
    let Some((cuda_branch, _)) = cuda_branch.split_once("#endif") else {
        return false;
    };
    amd_branch.contains("__builtin_trap();")
        && cuda_branch.contains("__trap();")
        && !helper.contains("return")
        && source.contains("if(decision_slot>u32_max_v1){resident_generation_trap_v1();return0;}")
        && source.contains("}resident_generation_trap_v1();return0;}")
        && source.matches("resident_generation_trap_v1();").count() == 2
        && !source.contains("asm(\"trap;\");")
}

#[test]
fn first_hit_keeps_one_explicit_32_lane_group_on_both_backends() {
    assert!(logical_subgroup_contract(FIRST_HIT));
}

#[test]
fn first_hit_rejects_a_narrow_mask_native_width_or_changed_launch() {
    for (before, after) in [
        ("unsigned long long mask", "unsigned mask"),
        ("0xffffffffull", "0xffffffffffffffffull"),
        ("offset, FIRST_HIT_LOGICAL_WIDTH_V1", "offset"),
        (
            "FIRST_HIT_LOGICAL_WIDTH_V1 = 32u",
            "FIRST_HIT_LOGICAL_WIDTH_V1 = 64u",
        ),
        (
            "event_count), FIRST_HIT_LOGICAL_WIDTH_V1>>>",
            "event_count), 64>>>",
        ),
    ] {
        assert!(
            FIRST_HIT.contains(before),
            "mutation must change actual source"
        );
        assert!(!logical_subgroup_contract(
            &FIRST_HIT.replace(before, after)
        ));
    }
}

#[test]
fn generation_rng_failure_keeps_platform_specific_hard_traps() {
    assert!(trap_contract(GENERATION));
}

#[test]
fn generation_rejects_missing_backend_traps_or_silent_rng_failure() {
    for (before, after) in [
        ("__builtin_trap();", "return;"),
        ("__trap();", "return;"),
        ("resident_generation_trap_v1();", "/* silently skipped */"),
    ] {
        assert!(
            GENERATION.contains(before),
            "mutation must change actual source"
        );
        assert!(!trap_contract(&GENERATION.replace(before, after)));
    }
}

fn population_callback_contract(source: &str) -> bool {
    let source = compact(source);
    let definition = concat!(
        "#ifdefined(__HIP_PLATFORM_AMD__)",
        "#defineNEO_POPULATION_HOST_CALLBACK_V1",
        "#else#defineNEO_POPULATION_HOST_CALLBACK_V1CUDART_CB#endif"
    );
    let Some((before, rest)) = source.split_once(definition) else {
        return false;
    };
    let Some((callbacks, after)) = rest.split_once("#undefNEO_POPULATION_HOST_CALLBACK_V1") else {
        return false;
    };
    ["gene", "scenario", "view"].into_iter().all(|kind| {
        callbacks.contains(&format!(
            "voidNEO_POPULATION_HOST_CALLBACK_V1release_{kind}_host_staging_v1(void*opaque)"
        ))
    }) && callbacks.matches("NEO_POPULATION_HOST_CALLBACK_V1").count() == 3
        && !before.contains("NEO_POPULATION_HOST_CALLBACK_V1")
        && !after.contains("NEO_POPULATION_HOST_CALLBACK_V1")
        && !source.contains("#defineCUDART_CB")
        && !source.contains("#undefCUDART_CB")
}

#[test]
fn population_host_callbacks_preserve_backend_abi_and_local_macro_scope() {
    assert!(population_callback_contract(POPULATION));
}

#[test]
fn population_rejects_wrong_callback_abi_missing_annotations_or_leaked_macro() {
    for (before, after) in [
        (
            "#if defined(__HIP_PLATFORM_AMD__)",
            "#if defined(__HIP_PLATFORM_NVIDIA__)",
        ),
        (
            "#define NEO_POPULATION_HOST_CALLBACK_V1\n",
            "#define NEO_POPULATION_HOST_CALLBACK_V1 __stdcall\n",
        ),
        (
            "#define NEO_POPULATION_HOST_CALLBACK_V1 CUDART_CB",
            "#define NEO_POPULATION_HOST_CALLBACK_V1",
        ),
        (
            "void NEO_POPULATION_HOST_CALLBACK_V1 release_gene",
            "void CUDART_CB release_gene",
        ),
        (
            "void NEO_POPULATION_HOST_CALLBACK_V1 release_scenario",
            "void CUDART_CB release_scenario",
        ),
        (
            "void NEO_POPULATION_HOST_CALLBACK_V1 release_view",
            "void CUDART_CB release_view",
        ),
        ("#undef NEO_POPULATION_HOST_CALLBACK_V1", ""),
    ] {
        assert!(
            POPULATION.contains(before),
            "mutation must change actual source"
        );
        assert!(!population_callback_contract(
            &POPULATION.replace(before, after)
        ));
    }
}

fn population_staging_retirement_contract(source: &str) -> bool {
    let source = compact(source);
    if !source.contains("#include\"resident_host_staging_v1.hpp\"") {
        return false;
    }
    for payload in [
        "GeneHostStagingV1",
        "ScenarioHostStagingV1",
        "ViewHostStagingV1",
    ] {
        if !source.contains(&format!(
            "neoethos_host_staging_v1::CallbackTicketV1<{payload}>::complete(opaque);"
        )) {
            return false;
        }
    }
    let Some((_, helper)) = source.split_once("std::int32_trelease_staging_after_stream_v1(")
    else {
        return false;
    };
    let Some((helper, _)) = helper.split_once("std::int32_tcopy_to_device(") else {
        return false;
    };
    if !helper.contains("neoethos_host_staging_v1::retire_after_stream_v1(staging,")
        || !helper.contains("returncudaLaunchHostFunc(stream,release,ticket)==cudaSuccess;")
        || !helper.contains("returncudaStreamSynchronize(stream)==cudaSuccess;")
        || helper.contains("deletestaging;")
        || !helper.contains("caseRetirementResultV1::Queued:returnNEO_POPULATION_STATUS_OK;")
        || !helper.contains("caseRetirementResultV1::TicketAllocationFailed:returnNEO_POPULATION_STATUS_ALLOCATION_FAILED;")
        || !helper.contains("caseRetirementResultV1::SubmissionFailed:returnNEO_POPULATION_STATUS_TRANSFER_FAILED;")
    {
        return false;
    }
    for (kind, next) in [
        ("genes", "scenarios"),
        ("scenarios", "resident_scenarios_v2"),
    ] {
        let start = format!("extern\"C\"std::int32_tneoethos_gpu_cuda_population_upload_{kind}(");
        let end = format!("extern\"C\"std::int32_tneoethos_gpu_cuda_population_upload_{next}(");
        let Some((_, upload)) = source.split_once(&start) else {
            return false;
        };
        let Some((upload, _)) = upload.split_once(&end) else {
            return false;
        };
        let callback = if kind == "genes" { "gene" } else { "scenario" };
        let retire = format!(
            "release_staging_after_stream_v1(session->stream,staging,release_{callback}_host_staging_v1)"
        );
        // One failed-copy retirement preserves the original status. The other
        // call supplies the normal success-path status. Neither may raw-free
        // staging after an unchecked wait.
        if upload.matches(&retire).count() != 2
            || upload.matches(&format!("status={retire}")).count() != 1
            || !upload.contains(&format!(
                "{retire};session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;returnstatus;"
            ))
            || upload.contains("cudaStreamSynchronize(")
        {
            return false;
        }
    }
    source.contains(
        "release_staging_after_stream_v1(session->stream,staging,release_view_host_staging_v1)",
    ) && source.contains("returncopy_status==NEO_POPULATION_STATUS_OK?release_status:copy_status;")
}

#[test]
fn population_all_staging_paths_use_shared_release_once_retirement() {
    assert!(population_staging_retirement_contract(POPULATION));
}

#[test]
fn population_rejects_raw_cleanup_ticket_mismatch_or_lost_copy_errors() {
    for (before, after) in [
        ("#include \"resident_host_staging_v1.hpp\"", ""),
        (
            "neoethos_host_staging_v1::CallbackTicketV1<GeneHostStagingV1>::complete(opaque);",
            "delete static_cast<GeneHostStagingV1*>(opaque);",
        ),
        (
            "cudaLaunchHostFunc(stream, release, ticket)",
            "cudaLaunchHostFunc(stream, release, staging)",
        ),
        (
            "cudaStreamSynchronize(stream) == cudaSuccess",
            "cudaStreamSynchronize(stream) != cudaSuccess",
        ),
        (
            "release_gene_host_staging_v1);\n    session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;\n    return status;",
            "release_gene_host_staging_v1);\n    session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;\n    return NEO_POPULATION_STATUS_OK;",
        ),
        (
            "release_scenario_host_staging_v1);\n    session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;\n    return status;",
            "release_scenario_host_staging_v1);\n    session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;\n    return NEO_POPULATION_STATUS_OK;",
        ),
        (
            "return copy_status == NEO_POPULATION_STATUS_OK ? release_status : copy_status;",
            "return release_status;",
        ),
    ] {
        assert!(
            POPULATION.contains(before),
            "mutation must change actual source"
        );
        assert!(!population_staging_retirement_contract(
            &POPULATION.replace(before, after)
        ));
    }
}

fn population_upload_failure_state_contract(source: &str) -> bool {
    let source = compact(source);
    for (kind, next, replacement, flags) in [
        (
            "genes",
            "scenarios",
            "if(!device_free_checked(session->candidate_ids)",
            &[
                "session->has_genes=false;",
                "session->population=0;",
                "session->gene_upload_bytes=0ull;",
                "session->uses_resident_gene_view_v2=false;",
            ][..],
        ),
        (
            "scenarios",
            "resident_scenarios_v2",
            "if(!session->release_scenarios()){",
            &[
                "session->has_scenarios=false;",
                "session->scenario_count=0;",
                "session->scenario_upload_bytes=0ull;",
                "session->metrics_ready=false;",
                "session->resident_canonical_base_scenarios_v3=false;",
            ][..],
        ),
    ] {
        let start = format!("extern\"C\"std::int32_tneoethos_gpu_cuda_population_upload_{kind}(");
        let end = format!("extern\"C\"std::int32_tneoethos_gpu_cuda_population_upload_{next}(");
        let Some((_, upload)) = source.split_once(&start) else {
            return false;
        };
        let Some((upload, _)) = upload.split_once(&end) else {
            return false;
        };
        let Some((before_replace, _)) = upload.split_once(replacement) else {
            return false;
        };
        let Some(staging) = before_replace.find("auto*staging=new") else {
            return false;
        };
        if flags.iter().any(|flag| before_replace.find(flag).is_none_or(|index| index < staging))
            || !upload.contains("if(status!=NEO_POPULATION_STATUS_OK){session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;returnstatus;}")
            || !upload.contains("if(strict_population_work_blocks_host_boundary_v1(session)){returnstrict_population_host_boundary_status_v1(session);}")
        {
            return false;
        }
        let release_failure = if kind == "genes" {
            "session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;returnNEO_POPULATION_STATUS_LAUNCH_FAILED;"
        } else {
            "if(!session->release_scenarios()){deletestaging;returnNEO_POPULATION_STATUS_LAUNCH_FAILED;}"
        };
        if !upload.contains(release_failure) {
            return false;
        }
    }
    let Some((_, scenario_release)) = source.split_once("boolrelease_scenarios(){") else {
        return false;
    };
    let Some((scenario_release, _)) = scenario_release.split_once("boolrelease_workspace(){")
    else {
        return false;
    };
    if !scenario_release.contains("if(!release_scenarios_checked_v2()){strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;returnfalse;}")
    {
        return false;
    }
    let Some((_, destroy)) =
        source.split_once("neoethos_gpu_cuda_population_destroy_terminal_checked_v2(")
    else {
        return false;
    };
    let Some((destroy, _)) =
        destroy.split_once("extern\"C\"voidneoethos_gpu_cuda_population_destroy(")
    else {
        return false;
    };
    let Some((guard, _)) = destroy.split_once("if(!session->release_terminal_checked_v2())") else {
        return false;
    };
    guard.contains("if(strict_population_work_blocks_host_boundary_v1(session)||")
        && guard.contains("session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;returnNEO_POPULATION_STATUS_INVALID_ARGUMENT;")
        && !guard.contains("deletesession;")
        && source.contains("session->strict_execution_state!=PopulationStrictExecutionStateV1::StrictIdle;")
        && source.contains("if(copy_status!=NEO_POPULATION_STATUS_OK||release_status!=NEO_POPULATION_STATUS_OK){session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;}")
}

#[test]
fn population_replacement_invalidates_old_success_and_quarantines_async_failure() {
    assert!(population_upload_failure_state_contract(POPULATION));
}

#[test]
fn population_rejects_stale_upload_success_or_reclaiming_poisoned_storage() {
    for (before, after) in [
        (
            "if (!device_free_checked(session->candidate_ids)",
            "if (device_free(session->candidate_ids)",
        ),
        ("if (!session->release_scenarios()) {", "if (false) {"),
        (
            "if (!release_scenarios_checked_v2()) {\n      strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;",
            "if (!release_scenarios_checked_v2()) {",
        ),
        ("session->has_genes = false;", "session->has_genes = true;"),
        ("session->population = 0;", "session->population = 1;"),
        ("session->gene_upload_bytes = 0ull;", ""),
        (
            "session->has_scenarios = false;",
            "session->has_scenarios = true;",
        ),
        ("session->scenario_count = 0;", ""),
        (
            "session->metrics_ready = false;",
            "session->metrics_ready = true;",
        ),
        (
            "session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;",
            "",
        ),
        (
            "session->strict_execution_state != PopulationStrictExecutionStateV1::StrictIdle",
            "false",
        ),
    ] {
        assert!(
            POPULATION.contains(before),
            "mutation must change actual source"
        );
        assert!(!population_upload_failure_state_contract(
            &POPULATION.replace(before, after)
        ));
    }
}

fn population_view_growth_failure_contract(source: &str) -> bool {
    let source = compact(source);
    let Some((_, growth)) = source.split_once("std::int32_tensure_device_capacity_v3(") else {
        return false;
    };
    let Some((growth, _)) = growth.split_once("}//namespace") else {
        return false;
    };
    if !growth.contains("if(!device_free_checked(*pointer)){returnNEO_POPULATION_STATUS_LAUNCH_FAILED;}*capacity=0;constautostatus=device_alloc(pointer,required);")
        || growth.contains("device_free(*pointer)")
    { return false; }
    let Some((_, view)) =
        source.split_once("extern\"C\"std::int32_tneoethos_gpu_cuda_population_bind_view_v1(")
    else {
        return false;
    };
    let Some((view, adaptive)) = view.split_once(
        "extern\"C\"std::int32_tneoethos_gpu_cuda_population_bind_resident_adaptive_view_v1(",
    ) else {
        return false;
    };
    let Some((adaptive, _)) = adaptive.split_once("#ifdefined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)")
    else {
        return false;
    };
    let Some((before_capacity, _)) = view.split_once("ensure_device_capacity_v3(") else {
        return false;
    };
    let Some((_, allocation)) =
        view.split_once("auto*staging=new(std::nothrow)ViewHostStagingV1{};")
    else {
        return false;
    };
    let Some((allocation, _)) = allocation.split_once("std::memcpy(") else {
        return false;
    };
    before_capacity.contains("constboolreplacing_view_storage=(view->view_kind==NEO_POPULATION_VIEW_ORDERED_INDICES&&rows>session->view_indices_capacity)||(view->adaptive_base_pips!=nullptr&&rows>session->adaptive_base_pips_capacity);")
        && view.contains("&session->view_indices_capacity,rows);if(status!=NEO_POPULATION_STATUS_OK){session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;returnstatus;}")
        && view.contains("&session->adaptive_base_pips_capacity,rows);if(status!=NEO_POPULATION_STATUS_OK){session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;returnstatus;}")
        && allocation.matches("if(replacing_view_storage){session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;}returnNEO_POPULATION_STATUS_ALLOCATION_FAILED;").count() == 2
        && adaptive.contains("constboolreplacing_adaptive_storage=rows>session->adaptive_base_pips_capacity;std::int32_tstatus=ensure_device_capacity_v3(")
        && adaptive.contains("&session->adaptive_base_pips_capacity,rows);if(status!=NEO_POPULATION_STATUS_OK){session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;returnstatus;}")
        && adaptive.contains("status=neoethos_gpu_cuda_population_bind_view_v1(session,view);if(status!=NEO_POPULATION_STATUS_OK){if(replacing_adaptive_storage){session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;}returnstatus;}")
        && adaptive.contains("if(cudaGetLastError()!=cudaSuccess){session->strict_execution_state=PopulationStrictExecutionStateV1::Poisoned;returnNEO_POPULATION_STATUS_LAUNCH_FAILED;}")
}

#[test]
fn population_view_growth_checks_release_and_quarantines_replaced_storage() {
    assert!(population_view_growth_failure_contract(POPULATION));
}

#[test]
fn population_rejects_unchecked_free_or_stale_view_after_growth_failure() {
    for (before, after) in [
        ("if (!device_free_checked(*pointer))", "if (false)"),
        ("rows > session->view_indices_capacity", "false"),
        ("rows > session->adaptive_base_pips_capacity", "false"),
        ("if (replacing_view_storage)", "if (false)"),
        ("if (replacing_adaptive_storage)", "if (false)"),
        (
            "session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;",
            "",
        ),
    ] {
        assert!(
            POPULATION.contains(before),
            "mutation must change actual source"
        );
        assert!(!population_view_growth_failure_contract(
            &POPULATION.replace(before, after)
        ));
    }
}
