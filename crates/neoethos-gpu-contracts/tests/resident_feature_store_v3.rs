use neoethos_gpu_contracts::resident_feature_store_v3::{
    CANONICAL_FEATURE_CONTENT_HASH_DOMAIN_V3, CANONICAL_MERKLE_CHUNK_ROWS_V3,
    CanonicalCudaSha256AuthorityV3, CuPqcHostCompilerV3, CuPqcSupportProbeV3,
    CudaPrimaryContextBuildIdentityV3, GpuOnlyResidentAdmissionRequestV3,
    GpuOnlyResidentAdmissionV3, ResidentFeatureContractErrorV3, ResidentFeatureLayoutRequestV3,
    ResidentFeatureLayoutV3, ResidentFeatureProducerV3, ResidentFeatureRouteV3,
    ResidentFeatureStageV3, ResidentParentDatasetLayoutV4, ResidentProducerCapabilityManifestV3,
    ResidentProducerCapabilityV3, ResidentReadyEventV3, ResidentValidityEncodingV3,
    ResidentWorkingSetBoundV3, ResidentWorkingSetRequestV3, SealedResidentFeatureStoreRequestV3,
    SealedResidentFeatureStoreV3, canonical_feature_merkle_sha256_host_oracle_v3,
    pack_logical_validity_u4_v3,
};

const HASH_A: [u8; 32] = [0x11; 32];
const HASH_B: [u8; 32] = [0x22; 32];
const HASH_C: [u8; 32] = [0x33; 32];
const HASH_D: [u8; 32] = [0x44; 32];

fn capability(producer: ResidentFeatureProducerV3) -> ResidentProducerCapabilityV3 {
    ResidentProducerCapabilityV3::new(
        producer,
        format!("neoethos.test.{}.resident.v3", producer.as_str()),
        HASH_A,
        format!("neoethos.test.{}.exact-bits.v3", producer.as_str()),
    )
    .expect("fixture capability must be valid")
}

fn complete_manifest() -> ResidentProducerCapabilityManifestV3 {
    ResidentProducerCapabilityManifestV3::seal(
        ResidentFeatureProducerV3::ALL
            .iter()
            .copied()
            .map(capability)
            .collect(),
    )
    .expect("complete ordered manifest must seal")
}

fn device() -> CudaPrimaryContextBuildIdentityV3 {
    CudaPrimaryContextBuildIdentityV3::new(
        0,
        [0x55; 16],
        8,
        6,
        HASH_A,
        "550.54.15",
        "12.8",
        "12.8",
        "sm_86",
        HASH_B,
        HASH_C,
        "neoethos.cuda-f64.exact-bits.v3",
    )
    .expect("fixture device identity must be valid")
}

fn routes() -> Vec<ResidentFeatureRouteV3> {
    vec![
        ResidentFeatureRouteV3::new(
            0,
            "classic_value",
            ResidentFeatureProducerV3::ClassicTa,
            Some("classic"),
            Some("value"),
            ResidentFeatureStageV3::Base,
            None,
            HASH_B,
            "classic.primary.f64",
            HASH_A,
        )
        .unwrap(),
        ResidentFeatureRouteV3::new(
            1,
            "classic_7_value",
            ResidentFeatureProducerV3::ClassicTa,
            Some("classic"),
            Some("value"),
            ResidentFeatureStageV3::Extended,
            Some(7),
            HASH_B,
            "classic.primary.f64",
            HASH_A,
        )
        .unwrap(),
    ]
}

fn working_set(
    device_free_bytes_snapshot: u64,
) -> Result<ResidentWorkingSetBoundV3, ResidentFeatureContractErrorV3> {
    ResidentWorkingSetRequestV3 {
        row_count: 4,
        column_count: 2,
        max_live_producer_bytes: 24,
        max_live_producer_scratch_bytes: 8,
        normalization_scratch_bytes: 16,
        fit_metadata_bytes: 8,
        pointer_and_schema_metadata_bytes: 16,
        device_free_bytes_snapshot,
        allocator_context_reserve_bytes: 0,
        reserve_policy_id: "neoethos.cuda.exact-allocator-context-reserve.v3".into(),
    }
    .seal()
}

fn admission() -> GpuOnlyResidentAdmissionV3 {
    GpuOnlyResidentAdmissionV3::seal(GpuOnlyResidentAdmissionRequestV3 {
        dataset_recipe_sha256: HASH_A,
        feature_plan_schema_sha256: HASH_B,
        route_plan_sha256: HASH_C,
        admission_identity_sha256: HASH_D,
        planned_routes: routes(),
        capabilities: complete_manifest(),
        device: device(),
        working_set: working_set(8_192).expect("fixture working set must fit"),
    })
    .expect("complete admission must seal")
}

fn layout(root: [u8; 32]) -> ResidentFeatureLayoutV3 {
    ResidentFeatureLayoutRequestV3 {
        row_count: 4,
        column_count: 2,
        canonical_content_merkle_sha256: root,
        source_column_count: 2,
        producer_batch_count: 2,
        validity_initialization_count: 1,
        value_layout_launch_count: 2,
        validity_boundary_launch_count: 2,
        layout_transform_value_bytes: 64,
        layout_transform_logical_validity_bytes: 8,
        full_feature_major_staging_bytes: 0,
        max_live_producer_bytes: 24,
        max_live_producer_scratch_bytes: 8,
        pre_materialization_free_bytes_snapshot: 8_192,
        post_parent_free_bytes_snapshot: 7_892,
        retained_parent_dataset_bytes: 300,
        remaining_peak_after_parent_bytes: 368,
        allocator_context_reserve_bytes: 0,
        reserve_policy_id: "neoethos.cuda.exact-allocator-context-reserve.v3".into(),
    }
    .seal()
    .expect("fixture layout must be valid")
}

fn parent_layout() -> ResidentParentDatasetLayoutV4 {
    ResidentParentDatasetLayoutV4::new(
        4, HASH_A, HASH_B, HASH_C, HASH_D, HASH_A, HASH_B, HASH_C, HASH_D, HASH_A,
    )
    .unwrap()
}

#[test]
fn cuda_build_identity_exposes_only_immutable_classic_ta_revalidation_evidence() {
    let identity = device();

    assert_eq!(identity.vector_ta_build_sha256(), HASH_B);
    assert_eq!(identity.nvcc_version(), "12.8");
    assert_eq!(
        identity.exact_math_authority(),
        "neoethos.cuda-f64.exact-bits.v3"
    );
}

#[test]
fn strict_admission_lists_every_missing_resident_producer_before_materialization() {
    let capabilities = ResidentProducerCapabilityManifestV3::seal(vec![capability(
        ResidentFeatureProducerV3::ClassicTa,
    )])
    .expect_err("one resident family must never admit strict end-to-end discovery");

    let ResidentFeatureContractErrorV3::MissingProducerCapabilities { missing } = capabilities
    else {
        panic!("unexpected error: {capabilities:?}");
    };
    assert_eq!(
        missing,
        ResidentFeatureProducerV3::ALL[1..],
        "the refusal must be complete, ordered, and deterministic"
    );
}

#[test]
fn strict_admission_has_no_caller_supports_gpu_boolean_or_partial_success() {
    let source = include_str!("../src/resident_feature_store_v3.rs");
    assert!(!source.contains("supports_gpu"));
    assert!(!source.contains("cpu_fallback"));
    assert!(!source.contains("allow_partial"));
    assert!(source.contains("MissingProducerCapabilities"));
    assert!(source.contains("ResidentFeatureProducerV3::ALL"));
}

#[test]
fn working_set_is_one_final_copy_plus_u4_and_max_live_batch_not_all_producers() {
    let bound = working_set(8_192).expect("small fixture must fit");
    assert_eq!(bound.final_bar_major_value_bytes(), 64);
    assert_eq!(bound.packed_validity_logical_bytes(), 4);
    assert_eq!(bound.packed_validity_allocated_bytes(), 4);
    assert_eq!(bound.parent_ohlcv_bytes(), 160);
    assert_eq!(bound.parent_clock_bytes(), 96);
    assert_eq!(bound.parent_smc_bytes(), 44);
    assert_eq!(bound.parent_dataset_bytes(), 300);
    assert_eq!(bound.canonical_root_bytes(), 32);
    assert_eq!(bound.active_view_indices_bytes(), 0);
    assert_eq!(bound.lazy_view_indices_capacity_bytes(), 0);
    assert_eq!(bound.merkle_leaf_count(), 3);
    assert_eq!(bound.merkle_scratch_bytes(), 192);
    assert_eq!(bound.max_live_producer_bytes(), 24);
    assert_eq!(bound.fit_metadata_bytes(), 8);
    assert_eq!(bound.full_feature_major_staging_bytes(), 0);
    assert_eq!(bound.steady_device_bytes(), 408);
    assert_eq!(bound.peak_device_bytes(), 668);
    assert_eq!(bound.remaining_peak_after_parent_bytes(), 368);
    assert!(bound.peak_device_bytes() <= bound.available_device_bytes());

    assert!(matches!(
        ResidentWorkingSetRequestV3 {
            row_count: usize::MAX,
            column_count: usize::MAX,
            max_live_producer_bytes: 0,
            max_live_producer_scratch_bytes: 0,
            normalization_scratch_bytes: 0,
            fit_metadata_bytes: 0,
            pointer_and_schema_metadata_bytes: 0,
            device_free_bytes_snapshot: u64::MAX,
            allocator_context_reserve_bytes: 0,
            reserve_policy_id: "test".into(),
        }
        .seal(),
        Err(ResidentFeatureContractErrorV3::ArithmeticOverflow { .. })
    ));
    assert!(matches!(
        working_set(667),
        Err(ResidentFeatureContractErrorV3::WorkingSetExceedsDevice { .. })
    ));
}

#[test]
fn large_matrix_receipt_never_claims_nominal_vram_or_population_capacity() {
    const GIB: u64 = 1 << 30;
    let feature_store = ResidentWorkingSetRequestV3 {
        row_count: 5_270_000,
        column_count: 257,
        max_live_producer_bytes: 0,
        max_live_producer_scratch_bytes: 0,
        normalization_scratch_bytes: 0,
        fit_metadata_bytes: 0,
        pointer_and_schema_metadata_bytes: 0,
        device_free_bytes_snapshot: 32 * GIB,
        allocator_context_reserve_bytes: 0,
        reserve_policy_id: "neoethos.cuda.exact-allocator-context-reserve.v3".into(),
    }
    .seal()
    .expect("32 GiB fixture must fit the feature store itself");

    assert_eq!(feature_store.final_bar_major_value_bytes(), 10_835_120_000);
    assert_eq!(feature_store.packed_validity_logical_bytes(), 677_195_000);
    assert_eq!(feature_store.parent_dataset_bytes(), 395_250_000);
    assert_eq!(feature_store.steady_device_bytes(), 11_907_565_032);
    assert!(feature_store.steady_device_bytes() > 11 * GIB);
    assert!(feature_store.steady_device_bytes() < 12 * GIB);

    // This receipt covers the immutable resident feature/parent store only.
    // Adaptive/gap/view buffers, the native session, and every GA/population
    // workspace are separate downstream charges. Therefore nominal 12 GiB
    // cannot imply an end-to-end fit, and no 16,384-population claim belongs
    // to this type even though the one-copy store itself is below 12 GiB.
    assert!(12 * GIB - feature_store.steady_device_bytes() < GIB);
    assert_eq!(feature_store.active_view_indices_bytes(), 0);
    assert_eq!(feature_store.lazy_view_indices_capacity_bytes(), 0);
}

#[test]
fn post_parent_snapshot_checks_only_the_remaining_peak_not_false_snapshot_equality() {
    let layout = layout(HASH_A);
    assert_eq!(layout.pre_materialization_free_bytes_snapshot(), 8_192);
    assert_eq!(layout.post_parent_free_bytes_snapshot(), 7_892);
    assert_eq!(layout.retained_parent_dataset_bytes(), 300);
    assert_eq!(layout.remaining_peak_after_parent_bytes(), 368);
    assert_eq!(layout.allocator_context_reserve_bytes(), 0);
    assert_eq!(
        layout.reserve_policy_id(),
        "neoethos.cuda.exact-allocator-context-reserve.v3"
    );

    let error = ResidentFeatureLayoutRequestV3 {
        row_count: 4,
        column_count: 2,
        canonical_content_merkle_sha256: HASH_A,
        source_column_count: 2,
        producer_batch_count: 2,
        validity_initialization_count: 1,
        value_layout_launch_count: 2,
        validity_boundary_launch_count: 2,
        layout_transform_value_bytes: 64,
        layout_transform_logical_validity_bytes: 8,
        full_feature_major_staging_bytes: 0,
        max_live_producer_bytes: 24,
        max_live_producer_scratch_bytes: 8,
        pre_materialization_free_bytes_snapshot: 8_192,
        post_parent_free_bytes_snapshot: 367,
        retained_parent_dataset_bytes: 300,
        remaining_peak_after_parent_bytes: 368,
        allocator_context_reserve_bytes: 0,
        reserve_policy_id: "neoethos.cuda.exact-allocator-context-reserve.v3".into(),
    }
    .seal()
    .expect_err("post-parent free bytes cannot underfund the remaining peak");
    assert!(matches!(
        error,
        ResidentFeatureContractErrorV3::WorkingSetExceedsDevice { .. }
    ));
}

#[test]
fn phase_one_binds_ordered_routes_native_sass_and_exact_working_set() {
    let admission = admission();
    assert_eq!(admission.planned_routes().len(), 2);
    assert_eq!(
        admission.planned_routes()[0].feature_name(),
        "classic_value"
    );
    assert_eq!(admission.planned_routes()[1].swept_period(), Some(7));
    assert_eq!(admission.device().ordinal(), 0);
    assert_eq!(admission.device().primary_context_process_token(), HASH_A);
    assert_eq!(admission.device().native_sass_target(), "sm_86");
    assert_eq!(admission.working_set().row_count(), 4);
    assert_eq!(admission.working_set().column_count(), 2);
    assert_eq!(admission.admission_identity_sha256(), HASH_D);

    assert!(matches!(
        CudaPrimaryContextBuildIdentityV3::new(
            0, [0x55; 16], 8, 6, HASH_A, "driver", "runtime", "nvcc", "sm_89", HASH_B, HASH_C,
            "exact",
        ),
        Err(ResidentFeatureContractErrorV3::NativeSassTargetMismatch { .. })
    ));
}

#[test]
fn phase_two_binds_one_honest_incremental_layout_without_raw_handles() {
    let admission = admission();
    let ready = ResidentReadyEventV3::new(0, HASH_A, HASH_B, HASH_C, 17)
        .expect("fixture event must be valid");
    let sealed = SealedResidentFeatureStoreV3::seal(
        &admission,
        SealedResidentFeatureStoreRequestV3 {
            admission_identity_sha256: HASH_D,
            final_feature_plan_v3_sha256: HASH_A,
            normalization_fit_sha256: HASH_B,
            source_provenance_sha256: HASH_C,
            ordered_feature_names: vec!["classic_value".into(), "classic_7_value".into()],
            layout: layout(HASH_D),
            parent_dataset: parent_layout(),
            ready_event: ready,
            sha256_authority: CanonicalCudaSha256AuthorityV3::portable_in_tree(),
        },
    )
    .expect("matching phase-two seal must succeed");

    assert_eq!(sealed.canonical_feature_content_merkle_sha256(), HASH_D);
    assert_eq!(sealed.layout().row_count(), 4);
    assert_eq!(sealed.layout().column_count(), 2);
    assert_eq!(sealed.layout().producer_batch_count(), 2);
    assert_eq!(sealed.layout().validity_initialization_count(), 1);
    assert_eq!(sealed.layout().layout_transform_launch_count(), 4);
    assert_eq!(sealed.layout().source_column_count(), 2);
    assert_eq!(sealed.layout().full_feature_major_staging_bytes(), 0);
    assert_eq!(
        sealed.layout().validity_encoding(),
        ResidentValidityEncodingV3::LosslessU4LogicalU8Sha256
    );
    assert_eq!(sealed.ready_event().host_synchronize_count(), 0);
    assert!(sealed.ready_event().consumer_must_wait_before_first_read());

    let source = include_str!("../src/resident_feature_store_v3.rs");
    assert!(!source.contains("pub raw_pointer"));
    assert!(!source.contains("pub device_ptr"));
    assert!(!source.contains("pub event_handle"));
    assert!(!source.contains("pub stream_handle"));
    assert!(!source.contains("d2d_transpose_count"));
}

#[test]
fn phase_two_refuses_device_context_schema_or_event_drift() {
    let admission = admission();
    let wrong_context = ResidentReadyEventV3::new(0, HASH_B, HASH_B, HASH_C, 17).unwrap();
    let error = SealedResidentFeatureStoreV3::seal(
        &admission,
        SealedResidentFeatureStoreRequestV3 {
            admission_identity_sha256: HASH_D,
            final_feature_plan_v3_sha256: HASH_A,
            normalization_fit_sha256: HASH_B,
            source_provenance_sha256: HASH_C,
            ordered_feature_names: vec!["classic_value".into(), "classic_7_value".into()],
            layout: layout(HASH_D),
            parent_dataset: parent_layout(),
            ready_event: wrong_context,
            sha256_authority: CanonicalCudaSha256AuthorityV3::portable_in_tree(),
        },
    )
    .expect_err("a different primary context must fail closed");
    assert!(matches!(
        error,
        ResidentFeatureContractErrorV3::PrimaryContextMismatch
    ));
}

#[test]
fn u4_validity_is_lossless_low_nibble_first_and_rejects_unknown_codes() {
    let all_codes = (0_u8..=9).collect::<Vec<_>>();
    assert_eq!(
        pack_logical_validity_u4_v3(&all_codes).unwrap(),
        vec![0x10, 0x32, 0x54, 0x76, 0x98]
    );
    assert_eq!(
        pack_logical_validity_u4_v3(&[9, 0, 7]).unwrap(),
        vec![0x09, 0x07],
        "the odd high nibble must be canonical zero"
    );
    assert!(matches!(
        pack_logical_validity_u4_v3(&[0, 10]),
        Err(ResidentFeatureContractErrorV3::InvalidValidityCode { code: 10, .. })
    ));
}

#[test]
fn u4_allocation_is_word_padded_for_every_partial_atomic_word() {
    for cells_mod_eight in 1_usize..=7 {
        let bound = ResidentWorkingSetRequestV3 {
            row_count: cells_mod_eight,
            column_count: 1,
            max_live_producer_bytes: 1,
            max_live_producer_scratch_bytes: 0,
            normalization_scratch_bytes: 0,
            fit_metadata_bytes: 0,
            pointer_and_schema_metadata_bytes: 0,
            device_free_bytes_snapshot: 16_384,
            allocator_context_reserve_bytes: 0,
            reserve_policy_id: "test".into(),
        }
        .seal()
        .unwrap();
        assert_eq!(
            bound.packed_validity_logical_bytes(),
            cells_mod_eight.div_ceil(2) as u64
        );
        assert_eq!(bound.packed_validity_allocated_bytes() % 4, 0);
        assert!(bound.packed_validity_allocated_bytes() >= bound.packed_validity_logical_bytes());
    }
}

#[test]
fn v3_merkle_oracle_preserves_every_f64_bit_and_logical_validity_code() {
    let timestamps = [1_i64, 2, 3];
    let names = vec!["a".to_owned(), "b".to_owned()];
    let bits = [
        0x8000_0000_0000_0000,
        f64::INFINITY.to_bits(),
        0x7ff8_0000_0000_0042,
        f64::NEG_INFINITY.to_bits(),
        1.0_f64.to_bits(),
        0x7ff0_0000_0000_0043,
    ];
    let logical_validity = [0_u8, 1, 2, 7, 8, 9];
    let packed = pack_logical_validity_u4_v3(&logical_validity).unwrap();
    let baseline =
        canonical_feature_merkle_sha256_host_oracle_v3(&timestamps, &names, &bits, &packed)
            .unwrap();

    let mut changed_bits = bits;
    changed_bits[2] ^= 1;
    assert_ne!(
        baseline,
        canonical_feature_merkle_sha256_host_oracle_v3(
            &timestamps,
            &names,
            &changed_bits,
            &packed,
        )
        .unwrap()
    );
    let mut changed_validity = logical_validity;
    changed_validity[5] = 8;
    assert_ne!(
        baseline,
        canonical_feature_merkle_sha256_host_oracle_v3(
            &timestamps,
            &names,
            &bits,
            &pack_logical_validity_u4_v3(&changed_validity).unwrap(),
        )
        .unwrap()
    );
    assert_ne!(
        baseline,
        canonical_feature_merkle_sha256_host_oracle_v3(&[1_i64, 2, 4], &names, &bits, &packed,)
            .unwrap()
    );
    assert_eq!(CANONICAL_MERKLE_CHUNK_ROWS_V3, 4096);
}

#[test]
fn v3_merkle_oracle_covers_even_and_odd_physical_u4_shapes() {
    for (rows, columns) in [(2_usize, 2_usize), (3, 1), (1, 3)] {
        let cells = rows * columns;
        let timestamps = (0..rows).map(|row| row as i64).collect::<Vec<_>>();
        let names = (0..columns)
            .map(|column| format!("f{column}"))
            .collect::<Vec<_>>();
        let bits = (0..cells)
            .map(|cell| (cell as f64).to_bits())
            .collect::<Vec<_>>();
        let logical = (0..cells).map(|cell| (cell % 10) as u8).collect::<Vec<_>>();
        let packed = pack_logical_validity_u4_v3(&logical).unwrap();
        assert_eq!(packed.len(), cells.div_ceil(2));
        if cells % 2 == 1 {
            assert_eq!(packed.last().copied().unwrap() & 0xf0, 0);
        }
        assert_ne!(
            canonical_feature_merkle_sha256_host_oracle_v3(&timestamps, &names, &bits, &packed,)
                .unwrap(),
            [0; 32]
        );
    }
}

#[test]
fn ready_event_contract_is_driver_runtime_interoperable_and_never_host_synchronizes() {
    let event = ResidentReadyEventV3::new(0, HASH_A, HASH_B, HASH_C, 99).unwrap();
    assert!(event.recorded_after_final_incremental_layout_normalization_and_merkle());
    assert!(event.consumer_must_wait_before_first_read());
    assert!(event.retains_store_until_consumer_completion());
    assert_eq!(event.host_synchronize_count(), 0);
    assert_eq!(
        event.interop_abi(),
        "cuda-driver-runtime.same-primary-context.cuEventRecord-cuStreamWaitEvent.v3"
    );
}

#[test]
fn canonical_content_domain_and_cupqc_portability_are_pinned() {
    assert_eq!(
        CANONICAL_FEATURE_CONTENT_HASH_DOMAIN_V3,
        b"neoethos.canonical-feature-content.merkle.root.v3\0"
    );
    assert!(CanonicalCudaSha256AuthorityV3::portable_in_tree().portable_path_is_mandatory());

    let linux_sm86 = CuPqcSupportProbeV3::new(
        "linux",
        "x86_64",
        CuPqcHostCompilerV3::Gcc,
        12,
        8,
        86,
        true,
        true,
        true,
        true,
    );
    assert!(linux_sm86.optional_acceleration_supported());

    let windows_sm86 = CuPqcSupportProbeV3::new(
        "windows",
        "x86_64",
        CuPqcHostCompilerV3::Msvc,
        12,
        8,
        86,
        true,
        true,
        true,
        true,
    );
    assert!(!windows_sm86.optional_acceleration_supported());

    let linux_sm120 = CuPqcSupportProbeV3::new(
        "linux",
        "x86_64",
        CuPqcHostCompilerV3::Gcc,
        12,
        8,
        120,
        true,
        true,
        true,
        true,
    );
    assert!(!linux_sm120.optional_acceleration_supported());
}
