use crate::population_auto_sizing_receipt_v1::{
    PopulationAutoCpuAuthorityV1, PopulationAutoSizingErrorCodeV1, PopulationAutoSizingRequestV1,
    PopulationAutoSizingRouteV1, quality_screen_candidate_chunk_v1,
    recompute_population_auto_receipt_identity_for_test_v1, seal_population_auto_sizing_receipt_v1,
    seal_population_auto_stage1_window_v1,
};

fn sha(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn gpu_request() -> PopulationAutoSizingRequestV1 {
    PopulationAutoSizingRequestV1 {
        population_auto: true,
        configured_population: 200,
        resident_parent_rows: 1_049_160,
        evaluation_rows: 262_290,
        feature_count: 1_800,
        month_capacity: 240,
        requested_max_indicators: 20,
        migration_enabled: false,
        parent_canonical_scope_identity_sha256: sha('1'),
        parent_dataset_identity_sha256: sha('2'),
        stage1_window: seal_population_auto_stage1_window_v1(
            &sha('2'),
            "selection_stage1",
            786_870,
            1_049_160,
        )
        .expect("stage1 window"),
        route: PopulationAutoSizingRouteV1::NativeCuda {
            selected_ordinal: 0,
            pre_parent_free_memory_bytes: 24 * 1024 * 1024 * 1024,
            cuda_device_identity_sha256: sha('4'),
            cuda_build_manifest_sha256: sha('5'),
            probe_receipt_identity_sha256: sha('6'),
        },
    }
}

#[test]
fn native_auto_binds_full_parent_and_stage1_time_extent_without_shrinking() {
    let receipt = seal_population_auto_sizing_receipt_v1(gpu_request()).expect("native receipt");
    receipt.validate().expect("self-validating receipt");
    assert_eq!(receipt.configured_population(), 200);
    assert!(receipt.resolved_population() >= 200);
    assert_eq!(receipt.resident_parent_rows(), 1_049_160);
    assert_eq!(receipt.evaluation_rows(), 262_290);
    assert_eq!(receipt.term_cap(), 20);
    assert_eq!(receipt.parent_device_bytes(), 15_187_640_160);
    assert_eq!(receipt.scenario_device_bytes_per_candidate(), 4_000);
    assert!(receipt.fixed_gene_capacity() >= receipt.resolved_population());
}

#[test]
fn auto_off_and_cpu_routes_are_typed_and_never_resize() {
    let mut request = gpu_request();
    request.population_auto = false;
    let disabled = seal_population_auto_sizing_receipt_v1(request).expect("disabled receipt");
    assert_eq!(disabled.resolved_population(), 200);
    assert_eq!(disabled.resolution_reason(), "auto_disabled");

    let mut request = gpu_request();
    request.route = PopulationAutoSizingRouteV1::CpuNoCompatibleGpu {
        authority: PopulationAutoCpuAuthorityV1::PhysicalGpuAbsence {
            platform: "linux".to_owned(),
            inventory_identity_sha256: sha('7'),
        },
    };
    let cpu = seal_population_auto_sizing_receipt_v1(request).expect("CPU receipt");
    assert_eq!(cpu.resolved_population(), 200);
    assert_eq!(cpu.resolution_reason(), "cpu_no_compatible_gpu");

    let mut request = gpu_request();
    request.route = PopulationAutoSizingRouteV1::CpuNoCompatibleGpu {
        authority: PopulationAutoCpuAuthorityV1::LegacyCudaZero {
            probe_receipt_identity_sha256: sha('8'),
        },
    };
    let legacy = seal_population_auto_sizing_receipt_v1(request).expect("legacy CPU receipt");
    let encoded = serde_json::to_vec(&legacy).expect("serialize legacy CPU receipt");
    let decoded: crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1 =
        serde_json::from_slice(&encoded).expect("deserialize legacy CPU receipt");
    decoded.validate().expect("legacy CPU receipt validates");
}

#[test]
fn configured_population_is_never_shrunk_by_time_occupancy_or_hard_caps() {
    let mut request = gpu_request();
    request.configured_population = 20_000;
    request.resident_parent_rows = 5_270_000;
    request.evaluation_rows = 5_270_000;
    request.feature_count = 1;
    request.stage1_window = seal_population_auto_stage1_window_v1(
        &request.parent_dataset_identity_sha256,
        "selection_stage1",
        0,
        request.evaluation_rows,
    )
    .expect("long stage1 window");
    let receipt = seal_population_auto_sizing_receipt_v1(request).expect("large configured fits");
    assert_eq!(receipt.resolved_population(), 20_000);
    assert!(receipt.occupancy_floor_overrode_time_target());
    assert_eq!(receipt.hard_growth_cap(), 16_384);
}

#[test]
fn configured_population_above_one_launch_cap_is_split_not_shrunk() {
    let mut request = gpu_request();
    request.configured_population = 1_000;
    request.resident_parent_rows = 1;
    request.evaluation_rows = 1;
    request.feature_count = 1;
    request.month_capacity = 10_000;
    request.route = PopulationAutoSizingRouteV1::NativeCuda {
        selected_ordinal: 0,
        pre_parent_free_memory_bytes: 200 * 1024 * 1024,
        cuda_device_identity_sha256: sha('4'),
        cuda_build_manifest_sha256: sha('5'),
        probe_receipt_identity_sha256: sha('6'),
    };
    request.stage1_window = seal_population_auto_stage1_window_v1(
        &request.parent_dataset_identity_sha256,
        "selection_stage1",
        0,
        1,
    )
    .expect("one-row stage1");
    let receipt = seal_population_auto_sizing_receipt_v1(request).expect("split-capable receipt");
    assert!(receipt.memory_population_cap() < 1_000);
    assert_eq!(receipt.resolved_population(), 1_000);
    assert_eq!(
        receipt.resolution_reason(),
        "native_cuda_configured_above_growth_cap_no_shrink"
    );
}

#[test]
fn configured_gene_or_one_scenario_no_room_fails_loudly() {
    let mut gene_no_room = gpu_request();
    gene_no_room.route = PopulationAutoSizingRouteV1::NativeCuda {
        selected_ordinal: 0,
        pre_parent_free_memory_bytes: 96 * 1024 * 1024,
        cuda_device_identity_sha256: sha('4'),
        cuda_build_manifest_sha256: sha('5'),
        probe_receipt_identity_sha256: sha('6'),
    };
    gene_no_room.resident_parent_rows = 1;
    gene_no_room.evaluation_rows = 1;
    gene_no_room.feature_count = 1;
    gene_no_room.configured_population = 1_000_000;
    gene_no_room.stage1_window = seal_population_auto_stage1_window_v1(
        &gene_no_room.parent_dataset_identity_sha256,
        "selection_stage1",
        0,
        1,
    )
    .expect("one-row stage1");
    let error = seal_population_auto_sizing_receipt_v1(gene_no_room).unwrap_err();
    assert_eq!(error.code(), PopulationAutoSizingErrorCodeV1::GeneNoRoom);

    let mut disabled_gene_no_room = gpu_request();
    disabled_gene_no_room.population_auto = false;
    disabled_gene_no_room.route = PopulationAutoSizingRouteV1::NativeCuda {
        selected_ordinal: 0,
        pre_parent_free_memory_bytes: 96 * 1024 * 1024,
        cuda_device_identity_sha256: sha('4'),
        cuda_build_manifest_sha256: sha('5'),
        probe_receipt_identity_sha256: sha('6'),
    };
    disabled_gene_no_room.resident_parent_rows = 1;
    disabled_gene_no_room.evaluation_rows = 1;
    disabled_gene_no_room.feature_count = 1;
    disabled_gene_no_room.configured_population = 1_000_000;
    disabled_gene_no_room.stage1_window = seal_population_auto_stage1_window_v1(
        &disabled_gene_no_room.parent_dataset_identity_sha256,
        "selection_stage1",
        0,
        1,
    )
    .expect("one-row stage1");
    let error = seal_population_auto_sizing_receipt_v1(disabled_gene_no_room).unwrap_err();
    assert_eq!(error.code(), PopulationAutoSizingErrorCodeV1::GeneNoRoom);

    let mut scenario_no_room = gpu_request();
    scenario_no_room.route = PopulationAutoSizingRouteV1::NativeCuda {
        selected_ordinal: 0,
        pre_parent_free_memory_bytes: 96 * 1024 * 1024,
        cuda_device_identity_sha256: sha('4'),
        cuda_build_manifest_sha256: sha('5'),
        probe_receipt_identity_sha256: sha('6'),
    };
    scenario_no_room.resident_parent_rows = 1;
    scenario_no_room.evaluation_rows = 1;
    scenario_no_room.feature_count = 1;
    scenario_no_room.configured_population = 1;
    scenario_no_room.month_capacity = u32::MAX as usize;
    scenario_no_room.stage1_window = seal_population_auto_stage1_window_v1(
        &scenario_no_room.parent_dataset_identity_sha256,
        "selection_stage1",
        0,
        1,
    )
    .expect("one-row stage1");
    let error = seal_population_auto_sizing_receipt_v1(scenario_no_room).unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::ScenarioNoRoom
    );
}

#[test]
fn term_cap_covers_five_term_templates_and_auto_rejects_unbounded_migration() {
    let mut request = gpu_request();
    request.requested_max_indicators = 1;
    assert_eq!(
        seal_population_auto_sizing_receipt_v1(request.clone())
            .expect("template floor")
            .term_cap(),
        5
    );
    request.feature_count = 3;
    assert_eq!(
        seal_population_auto_sizing_receipt_v1(request.clone())
            .expect("feature ceiling")
            .term_cap(),
        3
    );
    request.migration_enabled = true;
    let error = seal_population_auto_sizing_receipt_v1(request).unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::UnboundedMigrationTerms
    );

    let mut cpu_auto = gpu_request();
    cpu_auto.migration_enabled = true;
    cpu_auto.route = PopulationAutoSizingRouteV1::CpuNoCompatibleGpu {
        authority: PopulationAutoCpuAuthorityV1::LegacyCudaZero {
            probe_receipt_identity_sha256: sha('8'),
        },
    };
    let error = seal_population_auto_sizing_receipt_v1(cpu_auto).unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::UnboundedMigrationTerms
    );

    let mut cpu_disabled = gpu_request();
    cpu_disabled.population_auto = false;
    cpu_disabled.migration_enabled = true;
    cpu_disabled.route = PopulationAutoSizingRouteV1::CpuNoCompatibleGpu {
        authority: PopulationAutoCpuAuthorityV1::LegacyCudaZero {
            probe_receipt_identity_sha256: sha('8'),
        },
    };
    let error = seal_population_auto_sizing_receipt_v1(cpu_disabled).unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::UnboundedMigrationTerms
    );

    let mut disabled = gpu_request();
    disabled.population_auto = false;
    disabled.migration_enabled = true;
    let error = seal_population_auto_sizing_receipt_v1(disabled).unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::UnboundedMigrationTerms
    );
}

#[test]
fn quality_screen_chunks_before_every_unsplittable_gene_upload() {
    let receipt = seal_population_auto_sizing_receipt_v1(gpu_request()).expect("receipt");
    for mc_runs in [1usize, 100] {
        let chunk = quality_screen_candidate_chunk_v1(&receipt, 1_000_000, mc_runs, false, true)
            .expect("host MC chunk");
        assert!(chunk > 0);
        let uploaded_genes = chunk.checked_mul(mc_runs + 1).expect("test extent");
        assert!(uploaded_genes <= receipt.fixed_gene_capacity());
        assert!(chunk.saturating_mul(mc_runs) <= 131_072);
    }
    let device_chunk = quality_screen_candidate_chunk_v1(&receipt, 1_000_000, 100, true, true)
        .expect("device MC chunk");
    assert!(device_chunk <= receipt.fixed_gene_capacity());
    let error =
        quality_screen_candidate_chunk_v1(&receipt, 1_000_000, u32::MAX as usize, true, true)
            .unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::QualityScreenChunkNoRoom
    );
    assert_eq!(
        quality_screen_candidate_chunk_v1(&receipt, 10, 0, false, true).expect("cost-only chunk"),
        10
    );
    let zero_work_chunk = quality_screen_candidate_chunk_v1(&receipt, 10_000_000, 0, true, false)
        .expect("zero-work screen remains host bounded");
    assert!(zero_work_chunk <= 131_072);

    let mut cpu_request = gpu_request();
    cpu_request.population_auto = false;
    cpu_request.route = PopulationAutoSizingRouteV1::CpuNoCompatibleGpu {
        authority: PopulationAutoCpuAuthorityV1::LegacyCudaZero {
            probe_receipt_identity_sha256: sha('8'),
        },
    };
    let cpu = seal_population_auto_sizing_receipt_v1(cpu_request).expect("CPU receipt");
    let cpu_chunk = quality_screen_candidate_chunk_v1(&cpu, 1_000_000, 100, false, true)
        .expect("CPU quality screen remains host bounded without GPU capacity facts");
    assert!(cpu_chunk > 0);
    assert!(cpu_chunk.saturating_mul(100) <= 131_072);
}

#[test]
fn arithmetic_overflow_is_typed_and_never_saturates_or_panics() {
    let mut request = gpu_request();
    request.configured_population = usize::MAX;
    let error = seal_population_auto_sizing_receipt_v1(request).unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::ArithmeticOverflow
    );

    let mut request = gpu_request();
    request.feature_count = usize::MAX;
    request.requested_max_indicators = usize::MAX;
    let error = seal_population_auto_sizing_receipt_v1(request).unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::ArithmeticOverflow
    );

    let mut request = gpu_request();
    request.resident_parent_rows = usize::MAX;
    let error = seal_population_auto_sizing_receipt_v1(request).unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::ArithmeticOverflow
    );
}

#[test]
fn stage1_window_must_be_the_named_selection_view_inside_the_parent() {
    let mut request = gpu_request();
    request.stage1_window = seal_population_auto_stage1_window_v1(
        &request.parent_dataset_identity_sha256,
        "selection_stage1",
        request.resident_parent_rows,
        request.resident_parent_rows + 1,
    )
    .expect("well-formed but out-of-parent window");
    request.evaluation_rows = 1;
    let error = seal_population_auto_sizing_receipt_v1(request).unwrap_err();
    assert_eq!(error.code(), PopulationAutoSizingErrorCodeV1::InvalidInput);

    let mut request = gpu_request();
    request.stage1_window = seal_population_auto_stage1_window_v1(
        &request.parent_dataset_identity_sha256,
        "some_other_role",
        786_870,
        1_049_160,
    )
    .expect("well-formed wrong-role window");
    let error = seal_population_auto_sizing_receipt_v1(request).unwrap_err();
    assert_eq!(error.code(), PopulationAutoSizingErrorCodeV1::InvalidInput);
}

#[test]
fn receipt_roundtrip_and_every_bound_field_mutation_fail_identity_validation() {
    let receipt = seal_population_auto_sizing_receipt_v1(gpu_request()).expect("receipt");
    let encoded = serde_json::to_vec(&receipt).expect("serialize");
    let decoded: crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1 =
        serde_json::from_slice(&encoded).expect("deserialize");
    decoded.validate().expect("roundtrip validates");

    fn leaf_mutations(value: &serde_json::Value) -> Vec<(String, serde_json::Value)> {
        fn walk(
            value: &serde_json::Value,
            path: &str,
            output: &mut Vec<(String, serde_json::Value)>,
        ) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, child) in map {
                        let next = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{path}.{key}")
                        };
                        walk(child, &next, output);
                    }
                }
                serde_json::Value::Array(values) => {
                    for (index, child) in values.iter().enumerate() {
                        walk(child, &format!("{path}.{index}"), output);
                    }
                }
                serde_json::Value::Bool(value) => {
                    output.push((path.to_owned(), serde_json::Value::Bool(!value)));
                }
                serde_json::Value::Number(number) => {
                    let replacement = number
                        .as_u64()
                        .and_then(|value| value.checked_add(1))
                        .map(serde_json::Number::from)
                        .map(serde_json::Value::Number)
                        .unwrap_or_else(|| serde_json::json!(0));
                    output.push((path.to_owned(), replacement));
                }
                serde_json::Value::String(value) => {
                    output.push((path.to_owned(), serde_json::json!(format!("{value}x"))));
                }
                serde_json::Value::Null => {
                    output.push((path.to_owned(), serde_json::json!("not-null")));
                }
            }
        }
        let mut output = Vec::new();
        walk(value, "", &mut output);
        output
    }

    fn replace_path(value: &mut serde_json::Value, path: &str, replacement: serde_json::Value) {
        let mut parts = path.split('.').peekable();
        let mut cursor = value;
        while let Some(part) = parts.next() {
            if parts.peek().is_none() {
                if let serde_json::Value::Object(map) = cursor {
                    map.insert(part.to_owned(), replacement);
                    return;
                }
                if let serde_json::Value::Array(values) = cursor {
                    values[part.parse::<usize>().expect("array index")] = replacement;
                    return;
                }
                panic!("leaf parent is not a container: {path}");
            }
            cursor = match cursor {
                serde_json::Value::Object(map) => map.get_mut(part).expect("object path"),
                serde_json::Value::Array(values) => {
                    &mut values[part.parse::<usize>().expect("array index")]
                }
                _ => panic!("path crosses a scalar: {path}"),
            };
        }
    }

    let value = serde_json::to_value(&receipt).expect("value");
    for (path, replacement) in leaf_mutations(&value) {
        let mut tampered = value.clone();
        replace_path(&mut tampered, &path, replacement);
        let rejected = serde_json::from_value::<
            crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1,
        >(tampered)
        .map_or(true, |decoded| decoded.validate().is_err());
        assert!(rejected, "mutation {path} was accepted");
    }

    let mut unknown = value.clone();
    unknown["future_unreviewed_field"] = serde_json::json!(true);
    assert!(
        serde_json::from_value::<
            crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1,
        >(unknown)
        .is_err()
    );

    let mut missing = value;
    missing
        .as_object_mut()
        .expect("receipt object")
        .remove("term_cap");
    assert!(
        serde_json::from_value::<
            crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1,
        >(missing)
        .is_err()
    );
}

#[test]
fn recomputing_the_public_hash_cannot_forge_derived_plan_facts() {
    let receipt = seal_population_auto_sizing_receipt_v1(gpu_request()).expect("receipt");
    let mut value = serde_json::to_value(receipt).expect("value");
    value["resolved_population"] = serde_json::json!(201);
    value["resolution_reason"] = serde_json::json!("native_cuda_auto_grew");
    let mut forged: crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1 =
        serde_json::from_value(value).expect("shape-readable forged receipt");
    recompute_population_auto_receipt_identity_for_test_v1(&mut forged)
        .expect("attacker can recompute an unkeyed content hash");
    let error = forged.validate().unwrap_err();
    assert_eq!(
        error.code(),
        PopulationAutoSizingErrorCodeV1::InvalidReceipt
    );
}

#[test]
fn semantic_search_hash_binds_sizing_choices_but_not_the_hardware_snapshot() {
    let receipt = seal_population_auto_sizing_receipt_v1(gpu_request()).expect("receipt");
    let mut config = crate::discovery::DiscoveryConfig::default();
    config.population_auto = true;
    config.population = receipt.configured_population();
    config.max_indicators = 20;
    config.evaluation_symbol = "EURUSD".to_owned();
    config.evaluation_spread_pips = 1.2;
    config.evaluation_commission_per_trade = 7.0;
    config.target_profile.min_payoff_ratio = 0.0;
    config.runtime_overrides.stage1_window = crate::discovery::Stage1Window::MostRecent;
    let semantic_hash = crate::run_identity::population_auto_semantic_config_hash_for_v1(
        &config, &receipt, 10.0, false,
    )
    .expect("semantic hash");

    let mut hardware_mutation = gpu_request();
    hardware_mutation.route = PopulationAutoSizingRouteV1::NativeCuda {
        selected_ordinal: 7,
        pre_parent_free_memory_bytes: 25 * 1024 * 1024 * 1024,
        cuda_device_identity_sha256: sha('9'),
        cuda_build_manifest_sha256: sha('a'),
        probe_receipt_identity_sha256: sha('b'),
    };
    let hardware_receipt = seal_population_auto_sizing_receipt_v1(hardware_mutation)
        .expect("different hardware receipt");
    assert_eq!(
        hardware_receipt.resolved_population(),
        receipt.resolved_population()
    );
    assert_ne!(
        hardware_receipt.identity_sha256(),
        receipt.identity_sha256()
    );
    assert_eq!(
        crate::run_identity::population_auto_semantic_config_hash_for_v1(
            &config,
            &hardware_receipt,
            10.0,
            false,
        )
        .expect("hardware-independent semantic hash"),
        semantic_hash
    );
    let base_authority = crate::run_identity::build_population_auto_search_authority_v1(
        &config, &receipt, 10.0, false,
    )
    .expect("base authority");
    let hardware_authority = crate::run_identity::build_population_auto_search_authority_v1(
        &config,
        &hardware_receipt,
        10.0,
        false,
    )
    .expect("hardware authority");
    assert_ne!(base_authority, hardware_authority);
    assert!(
        base_authority
            .semantically_matches(&hardware_authority)
            .expect("typed semantic comparison")
    );

    let mut month_mutation = gpu_request();
    month_mutation.month_capacity = 241;
    let month_receipt =
        seal_population_auto_sizing_receipt_v1(month_mutation).expect("month mutation receipt");
    assert_eq!(
        month_receipt.resolved_population(),
        receipt.resolved_population()
    );
    assert_ne!(
        crate::run_identity::population_auto_semantic_config_hash_for_v1(
            &config,
            &month_receipt,
            10.0,
            false,
        )
        .expect("month-capacity semantic hash"),
        semantic_hash,
        "month capacity changes monthly scoring and therefore search semantics"
    );

    let mut configured_mutation = gpu_request();
    configured_mutation.configured_population = 201;
    let configured_receipt = seal_population_auto_sizing_receipt_v1(configured_mutation)
        .expect("configured mutation receipt");
    assert_eq!(
        configured_receipt.resolved_population(),
        receipt.resolved_population()
    );
    let mut configured_config = config.clone();
    configured_config.population = configured_receipt.configured_population();
    assert_ne!(
        crate::run_identity::population_auto_semantic_config_hash_for_v1(
            &configured_config,
            &configured_receipt,
            10.0,
            false,
        )
        .expect("configured mutation hash"),
        semantic_hash
    );

    let mut term_cap_mutation = gpu_request();
    term_cap_mutation.feature_count = 4;
    let term_cap_receipt = seal_population_auto_sizing_receipt_v1(term_cap_mutation)
        .expect("term-cap mutation receipt");
    assert_eq!(
        term_cap_receipt.resolved_population(),
        receipt.resolved_population()
    );
    assert_ne!(term_cap_receipt.term_cap(), receipt.term_cap());
    assert_ne!(
        crate::run_identity::population_auto_semantic_config_hash_for_v1(
            &config,
            &term_cap_receipt,
            10.0,
            false,
        )
        .expect("term-cap semantic hash"),
        semantic_hash,
        "receipt-derived term cap changes the searched gene space"
    );

    let mut stage_mutation = gpu_request();
    stage_mutation.stage1_window = seal_population_auto_stage1_window_v1(
        &stage_mutation.parent_dataset_identity_sha256,
        "selection_stage1",
        0,
        stage_mutation.evaluation_rows,
    )
    .expect("different exact stage1 window");
    let stage_receipt =
        seal_population_auto_sizing_receipt_v1(stage_mutation).expect("stage mutation receipt");
    let mut stage_config = config.clone();
    stage_config.runtime_overrides.stage1_window = crate::discovery::Stage1Window::Earliest;
    assert_ne!(
        crate::run_identity::population_auto_semantic_config_hash_for_v1(
            &stage_config,
            &stage_receipt,
            10.0,
            false,
        )
        .expect("stage mutation hash"),
        semantic_hash
    );
}

#[test]
fn strict_search_authority_rejects_missing_unknown_and_rehashed_inner_forgery() {
    let receipt = seal_population_auto_sizing_receipt_v1(gpu_request()).expect("receipt");
    let mut config = crate::discovery::DiscoveryConfig::default();
    config.population_auto = true;
    config.population = receipt.configured_population();
    config.max_indicators = 20;
    config.evaluation_symbol = "EURUSD".to_owned();
    config.evaluation_spread_pips = 1.2;
    config.evaluation_commission_per_trade = 7.0;
    config.target_profile.min_payoff_ratio = 0.5;
    config.runtime_overrides.stage1_window = crate::discovery::Stage1Window::MostRecent;
    let authority = crate::run_identity::build_population_auto_search_authority_v1(
        &config, &receipt, 10.0, false,
    )
    .expect("strict search authority");
    authority.validate().expect("authority validates");
    let mut wrong_requested = config.clone();
    wrong_requested.population += 1;
    assert!(
        crate::run_identity::build_population_auto_search_authority_v1(
            &wrong_requested,
            &receipt,
            10.0,
            false,
        )
        .is_err(),
        "requested population must exactly match the receipt's configured population"
    );

    let encoded = serde_json::to_value(&authority).expect("authority JSON");
    let decoded: crate::run_identity::PopulationAutoSearchAuthorityV1 =
        serde_json::from_value(encoded.clone()).expect("strict authority roundtrip");
    decoded.validate().expect("roundtrip validates");

    let mut missing = encoded.clone();
    missing
        .as_object_mut()
        .expect("authority object")
        .remove("population_auto_sizing_receipt");
    assert!(
        serde_json::from_value::<crate::run_identity::PopulationAutoSearchAuthorityV1>(missing)
            .is_err(),
        "required sizing receipt cannot default"
    );
    let mut unknown = encoded.clone();
    unknown["future_unreviewed_field"] = serde_json::json!(true);
    assert!(
        serde_json::from_value::<crate::run_identity::PopulationAutoSearchAuthorityV1>(unknown)
            .is_err(),
        "unknown authority fields fail closed"
    );
    let mut missing_inner = encoded.clone();
    missing_inner["resolved_config_stamp"]
        .as_object_mut()
        .expect("strict inner stamp")
        .remove("min_net_expectancy_per_trade");
    assert!(
        serde_json::from_value::<crate::run_identity::PopulationAutoSearchAuthorityV1>(
            missing_inner,
        )
        .is_err(),
        "inner decision fields cannot serde-default"
    );
    let mut unknown_inner = encoded.clone();
    unknown_inner["resolved_config_stamp"]["future_inner_field"] = serde_json::json!(true);
    assert!(
        serde_json::from_value::<crate::run_identity::PopulationAutoSearchAuthorityV1>(
            unknown_inner,
        )
        .is_err(),
        "unknown inner stamp fields fail closed"
    );
    for path in ["band_atr_pips", "session_spread_pips", "cost_band_pips"] {
        let mut missing_optional = encoded.clone();
        missing_optional["resolved_config_stamp"]
            .as_object_mut()
            .expect("strict inner stamp")
            .remove(path);
        assert!(
            serde_json::from_value::<crate::run_identity::PopulationAutoSearchAuthorityV1>(
                missing_optional,
            )
            .is_err(),
            "required optional field {path} cannot be omitted"
        );
    }
    for path in ["trailing_armed_floor_payoff", "trailing_ceiling_unmeasured"] {
        let mut missing_nested = encoded.clone();
        missing_nested["resolved_config_stamp"]["payoff_ceiling"]
            .as_object_mut()
            .expect("strict payoff ceiling")
            .remove(path);
        assert!(
            serde_json::from_value::<crate::run_identity::PopulationAutoSearchAuthorityV1>(
                missing_nested,
            )
            .is_err(),
            "required payoff field {path} cannot be omitted"
        );
    }
    let mut unknown_nested = encoded.clone();
    unknown_nested["resolved_config_stamp"]["payoff_ceiling"]["future_nested_field"] =
        serde_json::json!(true);
    assert!(
        serde_json::from_value::<crate::run_identity::PopulationAutoSearchAuthorityV1>(
            unknown_nested,
        )
        .is_err(),
        "unknown nested payoff fields fail closed"
    );

    let mut forged = encoded;
    forged["resolved_config_stamp"]["config_hash"] = serde_json::json!("fnv64:0000000000000001");
    let mut forged: crate::run_identity::PopulationAutoSearchAuthorityV1 =
        serde_json::from_value(forged).expect("shape-readable forgery");
    crate::run_identity::recompute_population_auto_search_authority_hash_for_test_v1(&mut forged)
        .expect("attacker can recompute an unkeyed outer hash");
    assert!(
        forged.validate().is_err(),
        "outer rehash cannot hide an invalid inner self-hash"
    );

    let mut contradictory = serde_json::to_value(&authority).expect("authority JSON");
    contradictory["resolved_config_stamp"]["spread_pips"] = serde_json::json!(99.0);
    let mut contradictory: crate::run_identity::PopulationAutoSearchAuthorityV1 =
        serde_json::from_value(contradictory).expect("shape-readable cost contradiction");
    crate::run_identity::recompute_resolved_config_stamp_hash_for_test_v2(&mut contradictory)
        .expect("attacker recomputes inner and outer hashes");
    assert!(
        contradictory.validate().is_err(),
        "raw spread/commission/pip fields must reconcile to stored round-trip cost"
    );

    for (name, field, value) in [
        (
            "reversed stop clamp",
            "sl_clamp_pips",
            serde_json::json!([20.0, 6.0]),
        ),
        (
            "reversed take-profit clamp",
            "tp_clamp_pips",
            serde_json::json!([45.0, 12.0]),
        ),
        (
            "reversed initializer RR band",
            "initializer_rr_max",
            serde_json::json!(1.0),
        ),
        (
            "nonpositive ATR scale",
            "band_atr_pips",
            serde_json::json!(0.0),
        ),
    ] {
        let mut forged_geometry =
            serde_json::to_value(&authority).expect("authority JSON for geometry forgery");
        forged_geometry["resolved_config_stamp"][field] = value;
        let mut forged_geometry: crate::run_identity::PopulationAutoSearchAuthorityV1 =
            serde_json::from_value(forged_geometry).expect("shape-readable geometry forgery");
        crate::run_identity::recompute_resolved_config_stamp_hash_for_test_v2(&mut forged_geometry)
            .expect("attacker recomputes inner and outer hashes");
        assert!(
            forged_geometry.validate().is_err(),
            "rehashed {name} must not validate as a search authority"
        );
    }
}

#[test]
fn direct_discovery_seals_the_run_owned_receipt_before_exact_search_without_reprobe() {
    let discovery = include_str!("discovery.rs");
    assert!(
        !discovery.contains("gpu_submission_ceiling("),
        "direct discovery must never re-probe free memory after sealed admission"
    );
    let begin_run = discovery
        .find("begin_exact_population_execution_run_v1")
        .expect("exact execution run begins");
    let exact_stage1 = discovery
        .find("let features_stage1 =")
        .expect("exact Stage1 view is resolved");
    let run_primitives = discovery
        .find("population_auto_sizing_primitives_v1")
        .expect("run-owned sizing primitives are borrowed");
    let receipt = discovery
        .find("seal_population_auto_sizing_receipt_v1")
        .expect("sizing receipt is sealed");
    let authority = discovery
        .find("build_population_auto_search_authority_v1")
        .expect("strict search authority is built");
    let exact_search = discovery
        .find("let search = evolve_search_with_progress_and_limits_exact")
        .expect("exact search is invoked");
    assert!(
        begin_run < exact_stage1
            && exact_stage1 < run_primitives
            && run_primitives < receipt
            && receipt < authority
            && authority < exact_search,
        "direct path ordering must be run -> exact Stage1 -> no-probe primitives -> receipt -> authority -> Search"
    );

    let search = include_str!("genetic/search_engine.rs");
    let exact_entry = search
        .split_once("pub(crate) fn evolve_search_with_progress_and_limits_exact")
        .expect("exact Search entry exists")
        .1;
    assert!(
        exact_entry.contains("PopulationAutoSearchAuthorityV1"),
        "exact Search must consume the strict authority, not only an unbound population"
    );
}
