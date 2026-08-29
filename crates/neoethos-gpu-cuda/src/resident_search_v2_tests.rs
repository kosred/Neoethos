use super::{
    ResidentSearchSlice2AdmissionErrorV2, ResidentSearchSlice2AdmissionOwnerV2,
    ResidentSearchSlice2AdmissionRequestV2, ResidentSearchSlice2AlignedFieldV2,
    ResidentSearchSlice2AlignedLayoutV2, ResidentSearchSlice2AllocationBudgetAxisV2,
    ResidentSearchSlice2AllocationCallV2, ResidentSearchSlice2AllocationCategoryV2,
    ResidentSearchSlice2AllocationFacadeV2, ResidentSearchSlice2AllocationSymbolV2,
    ResidentSearchSlice2AsyncAllocationArgsV2, ResidentSearchSlice2AuthorityBindingAxisV2,
    ResidentSearchSlice2AuthorityBindingV2, ResidentSearchSlice2CalibrationAxisV2,
    ResidentSearchSlice2CalibrationBindingV2, ResidentSearchSlice2GenerationReceiptV2,
    ResidentSearchSlice2HostAllocationArgsV2, ResidentSearchSlice2ObservedReserveAuthorityV2,
    ResidentSearchSlice2ObservedReserveSetV2, ResidentSearchSlice2ReceiptArithmeticV2,
    ResidentSearchSlice2ReceiptTotalAxisV2, ResidentSearchSlice2ReserveArithmeticV2,
    ResidentSearchSlice2ReserveAuthorityKindV2, ResidentSearchSlice2ReserveRelationV2,
    ResidentSearchSlice2ScoringArchiveReceiptV2, ResidentSearchSlice2ShapeAxisV2,
    ResidentSearchSlice2TrustedReserveAuthorityV2, ResidentSearchSlice2TrustedReserveSealV2,
    ResidentSearchSlice2TrustedReserveSetV2, admit_slice2_combined_fixture_v2,
    validate_and_seal_slice2_combined_v2,
};

const POPULATION_COUNT: u64 = 200;
const ARCHIVE_CAPACITY: u64 = 50_000;
const SIGNATURE_WORD_COUNT: u32 = 4;
const NOVELTY_NEIGHBOR_COUNT: u32 = 15;
const MAX_TERMS_PER_GENE: u32 = 16;
const TERMINAL_HOST_RECEIPT_BYTES: u64 = 104;
const TERMINAL_HOST_ALIGNMENT_BYTES: u64 = 8;
const SLICE2_ALIGNMENT_BYTES: u64 = 256;
const CUDA_HOST_ALLOC_PORTABLE: u32 = 0x01;
const REPLACEMENT_SUBTOTAL_BYTES: u64 = 23_707_648;
const GENERATION_TOTAL_BYTES: u64 = 241_408;
const SCORING_ARCHIVE_TOTAL_BYTES: u64 = 23_776_768;
const REQUESTED_DEVICE_SUM_BYTES: u64 = 24_018_176;
const ALLOCATOR_CONTEXT_HEADROOM_BYTES: u64 = 8_388_608;
const RETAINED_PRE_SEARCH_WORKSPACE_BYTES: u64 = 67_108_864;
const REMAINING_SEARCH_ALLOCATION_AFTER_TRIM_BYTES: u64 = 24_018_176;
const FULL_WORKSPACE_AUTHORITY_BYTES: u64 = 91_127_040;
const SAME_CONTEXT_FREE_BYTES: u64 = 32_406_784;
const DEVICE_UUID: [u8; 16] = [
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];
const PRIMARY_CONTEXT_IDENTITY: u64 = 0x2101;
const SEARCH_STREAM_IDENTITY: u64 = 0x2202;
const ACTIVE_POOL_IDENTITY: u64 = 0x2303;
const CUDA_BUILD_IDENTITY: u64 = 0x2404;
const KERNEL_SEMANTICS_IDENTITY: u64 = 0x2505;
const BINARY64_MATH_IDENTITY: u64 = 0x2606;
const PLAN_IDENTITY: u64 = 0x2707;
const RUN_IDENTITY: u64 = 0x2808;
const FULL_WORKSPACE_RECEIPT_IDENTITY: u64 = 0x2909;
const POST_TRIM_RECEIPT_IDENTITY: u64 = 0x2a0a;
const HEADROOM_AUTHORITY_IDENTITY: u64 = 0x3101;
const FULL_WORKSPACE_AUTHORITY_IDENTITY: u64 = 0x3202;
const RETAINED_AUTHORITY_IDENTITY: u64 = 0x3303;
const REMAINING_AUTHORITY_IDENTITY: u64 = 0x3404;
const SAME_CONTEXT_FREE_AUTHORITY_IDENTITY: u64 = 0x3505;
const PAIR_ALIAS_IDENTITY_V2: u64 = 0xa11a_5e00_0000_0001;

const R6_MUTATION_NAMES: &[&str] = &[
    "remove_missing_archive_validation",
    "remove_zero_archive_validation",
    "replace_named_layout_checks_with_subtotal_only",
    "workspace_partition_add_wrapping",
    "workspace_partition_add_saturating",
    "requested_device_sum_add_wrapping",
    "requested_device_sum_add_saturating",
    "same_context_free_minus_headroom_wrapping",
    "same_context_free_minus_headroom_saturating",
    "remove_allocator_context_headroom_identity_relation",
    "remove_full_workspace_authority_identity_relation",
    "remove_retained_pre_search_workspace_identity_relation",
    "remove_remaining_search_allocation_identity_relation",
    "remove_retained_plus_remaining_partition_relation",
    "remaining_budget_exact_fit_uses_strict_less_than",
    "same_context_budget_exact_fit_uses_strict_less_than",
    "accept_remaining_budget_one_byte_short",
    "accept_same_context_budget_one_byte_short",
    "remove_device_uuid_calibration_binding",
    "remove_primary_context_calibration_binding",
    "remove_search_stream_calibration_binding",
    "remove_active_pool_calibration_binding",
    "remove_cuda_build_calibration_binding",
    "remove_kernel_semantics_calibration_binding",
    "remove_binary64_math_calibration_binding",
    "remove_plan_calibration_binding",
    "remove_run_calibration_binding",
    "copy_declared_ledger_into_observed_state",
    "skip_terminal_host_receipt_call",
    "skip_generation_arena_call",
    "skip_scoring_archive_arena_call",
    "reorder_physical_allocation_calls",
    "change_terminal_host_receipt_ordinal",
    "change_generation_arena_ordinal",
    "change_scoring_archive_arena_ordinal",
    "change_terminal_host_receipt_symbol",
    "change_generation_arena_symbol",
    "change_scoring_archive_arena_symbol",
    "change_terminal_host_receipt_category",
    "change_generation_arena_category",
    "change_scoring_archive_arena_category",
    "change_terminal_host_receipt_requested_bytes",
    "change_generation_arena_requested_bytes",
    "change_scoring_archive_arena_requested_bytes",
    "change_terminal_host_receipt_aligned_bytes",
    "change_generation_arena_aligned_bytes",
    "change_scoring_archive_arena_aligned_bytes",
    "change_terminal_host_receipt_alignment",
    "change_generation_arena_alignment",
    "change_scoring_archive_arena_alignment",
    "change_terminal_host_receipt_flags",
    "change_generation_arena_flags",
    "change_scoring_archive_arena_flags",
    "change_terminal_host_receipt_stream",
    "change_generation_arena_stream",
    "change_scoring_archive_arena_stream",
    "change_terminal_host_receipt_pool",
    "change_generation_arena_pool",
    "change_scoring_archive_arena_pool",
    "prepend_extra_observed_entry",
    "append_extra_observed_entry",
    "allocate_while_queueing_generation_two",
    "allocate_while_queueing_generation_three",
    "trust_declared_replacement_subtotal",
    "trust_declared_generation_total",
    "trust_declared_scoring_archive_total",
    "replacement_subtotal_add_wrapping",
    "replacement_subtotal_add_saturating",
    "generation_total_add_wrapping",
    "generation_total_add_saturating",
    "scoring_archive_total_add_wrapping",
    "scoring_archive_total_add_saturating",
    "return_replacement_total_mismatch_before_overflow",
    "return_generation_total_mismatch_before_overflow",
    "return_scoring_archive_total_mismatch_before_overflow",
    "remove_allocator_context_headroom_expected_bytes",
    "remove_full_workspace_authority_expected_bytes",
    "remove_retained_pre_search_workspace_expected_bytes",
    "remove_remaining_search_allocation_expected_bytes",
    "remove_same_context_free_expected_bytes",
    "remove_allocator_context_headroom_full_binding",
    "remove_full_workspace_authority_full_binding",
    "remove_retained_pre_search_workspace_full_binding",
    "remove_remaining_search_allocation_full_binding",
    "remove_same_context_free_full_binding",
    "remove_reserve_binding_device_uuid",
    "remove_reserve_binding_primary_context",
    "remove_reserve_binding_search_stream",
    "remove_reserve_binding_active_pool",
    "remove_reserve_binding_run_identity",
    "remove_reserve_binding_full_workspace_receipt_identity",
    "remove_reserve_binding_post_trim_receipt_identity",
    "remove_reserve_binding_authority_identity",
    "accept_four_way_reserve_identity_alias",
    "accept_headroom_full_workspace_authority_identity_alias",
    "accept_headroom_retained_authority_identity_alias",
    "accept_headroom_remaining_authority_identity_alias",
    "accept_full_workspace_retained_authority_identity_alias",
    "accept_full_workspace_remaining_authority_identity_alias",
    "accept_retained_remaining_authority_identity_alias",
    "accept_coordinated_workspace_byte_substitution",
    "accept_coordinated_context_budget_byte_substitution",
    "accept_coordinated_reserve_binding_substitution",
    "truncate_reserve_binding_identities_to_u32",
    "compare_reserve_binding_uuid_byte_zero_only",
    "trust_terminal_declared_symbol_instead_of_host_method",
    "trust_generation_declared_symbol_instead_of_async_method",
    "trust_scoring_archive_declared_symbol_instead_of_async_method",
    "allocate_before_native_create",
    "remove_host_allocator_method_count",
    "remove_async_allocator_method_count",
    "truncate_calibration_identities_to_u32",
    "compare_calibration_uuid_byte_zero_only",
    "swap_allocator_context_headroom_and_full_workspace_precedence",
    "swap_full_workspace_and_retained_precedence",
    "validate_reserve_binding_before_bytes",
    "swap_device_uuid_and_primary_context_precedence",
    "expose_trusted_capability_fields",
    "derive_clone_for_trusted_capability_graph",
    "derive_copy_for_trusted_capability_graph",
    "derive_default_for_trusted_capability_graph",
    "add_raw_trusted_reserve_constructor",
    "add_raw_trusted_reserve_accessor",
    "pass_unsealed_trusted_reserve_set",
    "allow_trusted_fixture_minter_arguments",
    "ungate_trusted_fixture_minter",
    "omit_expected_calibration_from_trusted_seal",
    "omit_full_workspace_provenance_from_trusted_seal",
    "omit_post_trim_provenance_from_trusted_seal",
    "accept_coordinated_observed_and_plain_trusted_substitution",
    "remove_trusted_fixture_field_inspection",
    "remove_native_create_event_recording",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ResidentSearchSlice2RecorderPhaseV2 {
    #[default]
    BeforeNativeCreate,
    NativeCreateBegun,
    AllocationsComplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentSearchSlice2RecorderEventV2 {
    NativeCreate {
        phase_before: ResidentSearchSlice2RecorderPhaseV2,
    },
    Allocation {
        phase_at_call: ResidentSearchSlice2RecorderPhaseV2,
        call: ResidentSearchSlice2AllocationCallV2,
    },
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AllocationRecorderV2 {
    phase: ResidentSearchSlice2RecorderPhaseV2,
    native_create_count: u64,
    host_allocator_method_count: u64,
    async_allocator_method_count: u64,
    physical_allocator_count: u64,
    generation_arena_count: u64,
    scoring_archive_arena_count: u64,
    archive_only_arena_count: u64,
    chronology: Vec<ResidentSearchSlice2RecorderEventV2>,
    observed: Vec<ResidentSearchSlice2AllocationCallV2>,
}

impl ResidentSearchSlice2AllocationRecorderV2 {
    fn phase(&self) -> ResidentSearchSlice2RecorderPhaseV2 {
        self.phase
    }
    fn native_create_count(&self) -> u64 {
        self.native_create_count
    }
    fn host_allocator_method_count(&self) -> u64 {
        self.host_allocator_method_count
    }
    fn async_allocator_method_count(&self) -> u64 {
        self.async_allocator_method_count
    }
    fn physical_allocator_count(&self) -> u64 {
        self.physical_allocator_count
    }
    fn generation_arena_count(&self) -> u64 {
        self.generation_arena_count
    }
    fn scoring_archive_arena_count(&self) -> u64 {
        self.scoring_archive_arena_count
    }
    fn archive_only_arena_count(&self) -> u64 {
        self.archive_only_arena_count
    }
    fn chronology(&self) -> &[ResidentSearchSlice2RecorderEventV2] {
        &self.chronology
    }
    fn observed(&self) -> &[ResidentSearchSlice2AllocationCallV2] {
        &self.observed
    }
    fn snapshot(&self) -> Self {
        self.clone()
    }
}

impl ResidentSearchSlice2AllocationFacadeV2 for ResidentSearchSlice2AllocationRecorderV2 {
    fn begin_native_create(&mut self) {
        let phase_before = self.phase;
        self.native_create_count = self
            .native_create_count
            .checked_add(1)
            .expect("native-create count is bounded by the fixture");
        self.chronology
            .push(ResidentSearchSlice2RecorderEventV2::NativeCreate { phase_before });
        self.phase = ResidentSearchSlice2RecorderPhaseV2::NativeCreateBegun;
    }

    fn cuda_host_alloc(&mut self, actual: ResidentSearchSlice2HostAllocationArgsV2) {
        self.host_allocator_method_count = self
            .host_allocator_method_count
            .checked_add(1)
            .expect("host allocator method count is bounded by the fixture");
        let phase_at_call = self.phase;
        let call = ResidentSearchSlice2AllocationCallV2 {
            ordinal: actual.ordinal,
            symbol: ResidentSearchSlice2AllocationSymbolV2::CudaHostAlloc,
            category: actual.category,
            requested_bytes: actual.requested_bytes,
            aligned_bytes: actual.aligned_bytes,
            alignment_bytes: actual.alignment_bytes,
            flags: actual.flags,
            stream_identity: None,
            pool_identity: None,
        };
        self.physical_allocator_count = self
            .physical_allocator_count
            .checked_add(1)
            .expect("physical allocation count is bounded by the fixture");
        match call.category {
            ResidentSearchSlice2AllocationCategoryV2::GenerationArena => {
                self.generation_arena_count = self
                    .generation_arena_count
                    .checked_add(1)
                    .expect("generation allocation count is bounded by the fixture");
            }
            ResidentSearchSlice2AllocationCategoryV2::ScoringArchiveArena => {
                self.scoring_archive_arena_count = self
                    .scoring_archive_arena_count
                    .checked_add(1)
                    .expect("scoring/archive allocation count is bounded by the fixture");
            }
            ResidentSearchSlice2AllocationCategoryV2::ArchiveOnlyArena => {
                self.archive_only_arena_count = self
                    .archive_only_arena_count
                    .checked_add(1)
                    .expect("archive-only allocation count is bounded by the fixture");
            }
            ResidentSearchSlice2AllocationCategoryV2::TerminalHostReceipt => {}
        }
        self.chronology
            .push(ResidentSearchSlice2RecorderEventV2::Allocation {
                phase_at_call,
                call,
            });
        self.observed.push(call);
        if self.phase == ResidentSearchSlice2RecorderPhaseV2::NativeCreateBegun
            && self.physical_allocator_count == 3
        {
            self.phase = ResidentSearchSlice2RecorderPhaseV2::AllocationsComplete;
        }
    }

    fn cuda_malloc_async(&mut self, actual: ResidentSearchSlice2AsyncAllocationArgsV2) {
        self.async_allocator_method_count = self
            .async_allocator_method_count
            .checked_add(1)
            .expect("async allocator method count is bounded by the fixture");
        let phase_at_call = self.phase;
        let call = ResidentSearchSlice2AllocationCallV2 {
            ordinal: actual.ordinal,
            symbol: ResidentSearchSlice2AllocationSymbolV2::CudaMallocAsync,
            category: actual.category,
            requested_bytes: actual.requested_bytes,
            aligned_bytes: actual.aligned_bytes,
            alignment_bytes: actual.alignment_bytes,
            flags: actual.flags,
            stream_identity: Some(actual.stream_identity),
            pool_identity: Some(actual.pool_identity),
        };
        self.physical_allocator_count = self
            .physical_allocator_count
            .checked_add(1)
            .expect("physical allocation count is bounded by the fixture");
        match call.category {
            ResidentSearchSlice2AllocationCategoryV2::GenerationArena => {
                self.generation_arena_count = self
                    .generation_arena_count
                    .checked_add(1)
                    .expect("generation allocation count is bounded by the fixture");
            }
            ResidentSearchSlice2AllocationCategoryV2::ScoringArchiveArena => {
                self.scoring_archive_arena_count = self
                    .scoring_archive_arena_count
                    .checked_add(1)
                    .expect("scoring/archive allocation count is bounded by the fixture");
            }
            ResidentSearchSlice2AllocationCategoryV2::ArchiveOnlyArena => {
                self.archive_only_arena_count = self
                    .archive_only_arena_count
                    .checked_add(1)
                    .expect("archive-only allocation count is bounded by the fixture");
            }
            ResidentSearchSlice2AllocationCategoryV2::TerminalHostReceipt => {}
        }
        self.chronology
            .push(ResidentSearchSlice2RecorderEventV2::Allocation {
                phase_at_call,
                call,
            });
        self.observed.push(call);
        if self.phase == ResidentSearchSlice2RecorderPhaseV2::NativeCreateBegun
            && self.physical_allocator_count == 3
        {
            self.phase = ResidentSearchSlice2RecorderPhaseV2::AllocationsComplete;
        }
    }
}

fn checked_sum(values: &[u64]) -> u64 {
    values
        .iter()
        .copied()
        .try_fold(0_u64, u64::checked_add)
        .expect("fixture byte sum must fit")
}

fn slice2_layout_subtotal(layout: &ResidentSearchSlice2AlignedLayoutV2) -> u64 {
    checked_sum(&[
        layout.archive_gene_scalars,
        layout.archive_term_indices,
        layout.archive_term_weights,
        layout.archive_metric_rows,
        layout.archive_signatures,
        layout.archive_hashes,
        layout.current_population_signatures,
        layout.novelty_scores,
        layout.exact_top_k_keys,
        layout.admission_flags,
        layout.admission_offsets,
        layout.archive_control_and_seal,
    ])
}

fn archive_subreceipt_bytes(layout: &ResidentSearchSlice2AlignedLayoutV2) -> u64 {
    checked_sum(&[
        layout.archive_gene_scalars,
        layout.archive_term_indices,
        layout.archive_term_weights,
        layout.archive_metric_rows,
        layout.archive_signatures,
        layout.archive_hashes,
    ])
}

fn valid_slice2_layout() -> ResidentSearchSlice2AlignedLayoutV2 {
    let mut layout = ResidentSearchSlice2AlignedLayoutV2 {
        archive_gene_scalars: 3_600_128,
        archive_term_indices: 6_400_000,
        archive_term_weights: 6_400_000,
        archive_metric_rows: 5_200_128,
        archive_signatures: 1_600_000,
        archive_hashes: 400_128,
        current_population_signatures: 6_400,
        novelty_scores: 1_792,
        exact_top_k_keys: 96_000,
        admission_flags: 1_024,
        admission_offsets: 1_792,
        archive_control_and_seal: 256,
        replacement_subtotal_bytes: 0,
    };
    layout.replacement_subtotal_bytes = slice2_layout_subtotal(&layout);
    assert_eq!(
        layout.replacement_subtotal_bytes,
        REPLACEMENT_SUBTOTAL_BYTES
    );
    layout
}

fn valid_generation_receipt() -> ResidentSearchSlice2GenerationReceiptV2 {
    let components = [
        14_592, 25_600, 25_600, 65_792, 20_992, 8_192, 5_120, 9_472, 65_536, 256, 256,
    ];
    let receipt = ResidentSearchSlice2GenerationReceiptV2 {
        logical_gene_scalar_bytes: components[0],
        logical_gene_index_bytes: components[1],
        logical_gene_weight_bytes: components[2],
        offspring_bytes: components[3],
        metric_row_bytes: components[4],
        rank_key_bytes: components[5],
        selection_bytes: components[6],
        dedup_hash_bytes: components[7],
        cub_scratch_bytes: components[8],
        retained_evaluation_workspace_bytes: components[9],
        terminal_device_receipt_bytes: components[10],
        total_device_bytes: checked_sum(&components),
    };
    assert_eq!(receipt.total_device_bytes, GENERATION_TOTAL_BYTES);
    receipt
}

fn generation_component_sum(receipt: &ResidentSearchSlice2GenerationReceiptV2) -> u64 {
    checked_sum(&[
        receipt.logical_gene_scalar_bytes,
        receipt.logical_gene_index_bytes,
        receipt.logical_gene_weight_bytes,
        receipt.offspring_bytes,
        receipt.metric_row_bytes,
        receipt.rank_key_bytes,
        receipt.selection_bytes,
        receipt.dedup_hash_bytes,
        receipt.cub_scratch_bytes,
        receipt.retained_evaluation_workspace_bytes,
        receipt.terminal_device_receipt_bytes,
    ])
}

fn scoring_archive_receipt(
    layout: ResidentSearchSlice2AlignedLayoutV2,
) -> ResidentSearchSlice2ScoringArchiveReceiptV2 {
    let fitness_score_bytes = 1_792;
    let decision_key_bytes = 1_792;
    let cub_scratch_bytes = 65_536;
    let total_device_bytes = checked_sum(&[
        fitness_score_bytes,
        decision_key_bytes,
        cub_scratch_bytes,
        layout.archive_gene_scalars,
        layout.archive_term_indices,
        layout.archive_term_weights,
        layout.archive_metric_rows,
        layout.archive_signatures,
        layout.archive_hashes,
        layout.current_population_signatures,
        layout.novelty_scores,
        layout.exact_top_k_keys,
        layout.admission_flags,
        layout.admission_offsets,
        layout.archive_control_and_seal,
    ]);
    ResidentSearchSlice2ScoringArchiveReceiptV2 {
        fitness_score_bytes,
        decision_key_bytes,
        cub_scratch_bytes,
        layout,
        total_device_bytes,
    }
}

fn observed_authority_binding(authority_identity: u64) -> ResidentSearchSlice2AuthorityBindingV2 {
    ResidentSearchSlice2AuthorityBindingV2 {
        device_uuid: DEVICE_UUID,
        primary_context_identity: PRIMARY_CONTEXT_IDENTITY,
        search_stream_identity: SEARCH_STREAM_IDENTITY,
        active_pool_identity: ACTIVE_POOL_IDENTITY,
        run_identity: RUN_IDENTITY,
        full_workspace_receipt_identity: FULL_WORKSPACE_RECEIPT_IDENTITY,
        post_trim_receipt_identity: POST_TRIM_RECEIPT_IDENTITY,
        authority_identity,
    }
}

fn observed_authority(
    bytes: u64,
    authority_identity: u64,
) -> ResidentSearchSlice2ObservedReserveAuthorityV2 {
    ResidentSearchSlice2ObservedReserveAuthorityV2 {
        bytes,
        binding: observed_authority_binding(authority_identity),
    }
}

fn valid_observed_reserve() -> ResidentSearchSlice2ObservedReserveSetV2 {
    ResidentSearchSlice2ObservedReserveSetV2 {
        allocator_context_headroom: observed_authority(
            ALLOCATOR_CONTEXT_HEADROOM_BYTES,
            HEADROOM_AUTHORITY_IDENTITY,
        ),
        full_workspace_authority: observed_authority(
            FULL_WORKSPACE_AUTHORITY_BYTES,
            FULL_WORKSPACE_AUTHORITY_IDENTITY,
        ),
        retained_pre_search_workspace: observed_authority(
            RETAINED_PRE_SEARCH_WORKSPACE_BYTES,
            RETAINED_AUTHORITY_IDENTITY,
        ),
        remaining_search_allocation_after_trim: observed_authority(
            REMAINING_SEARCH_ALLOCATION_AFTER_TRIM_BYTES,
            REMAINING_AUTHORITY_IDENTITY,
        ),
        same_context_free: observed_authority(
            SAME_CONTEXT_FREE_BYTES,
            SAME_CONTEXT_FREE_AUTHORITY_IDENTITY,
        ),
    }
}

fn valid_calibration() -> ResidentSearchSlice2CalibrationBindingV2 {
    ResidentSearchSlice2CalibrationBindingV2 {
        device_uuid: DEVICE_UUID,
        primary_context_identity: PRIMARY_CONTEXT_IDENTITY,
        search_stream_identity: SEARCH_STREAM_IDENTITY,
        active_pool_identity: ACTIVE_POOL_IDENTITY,
        cuda_build_identity: CUDA_BUILD_IDENTITY,
        kernel_semantics_identity: KERNEL_SEMANTICS_IDENTITY,
        binary64_math_identity: BINARY64_MATH_IDENTITY,
        plan_identity: PLAN_IDENTITY,
        run_identity: RUN_IDENTITY,
    }
}

#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
fn mint_r6_trusted_reserve_seal_for_fixture_v2() -> ResidentSearchSlice2TrustedReserveSealV2 {
    let trusted_binding = |authority_identity| ResidentSearchSlice2AuthorityBindingV2 {
        device_uuid: DEVICE_UUID,
        primary_context_identity: PRIMARY_CONTEXT_IDENTITY,
        search_stream_identity: SEARCH_STREAM_IDENTITY,
        active_pool_identity: ACTIVE_POOL_IDENTITY,
        run_identity: RUN_IDENTITY,
        full_workspace_receipt_identity: FULL_WORKSPACE_RECEIPT_IDENTITY,
        post_trim_receipt_identity: POST_TRIM_RECEIPT_IDENTITY,
        authority_identity,
    };
    let trusted_reserve = ResidentSearchSlice2TrustedReserveSetV2 {
        allocator_context_headroom: ResidentSearchSlice2TrustedReserveAuthorityV2 {
            expected_bytes: ALLOCATOR_CONTEXT_HEADROOM_BYTES,
            expected_binding: trusted_binding(HEADROOM_AUTHORITY_IDENTITY),
        },
        full_workspace_authority: ResidentSearchSlice2TrustedReserveAuthorityV2 {
            expected_bytes: FULL_WORKSPACE_AUTHORITY_BYTES,
            expected_binding: trusted_binding(FULL_WORKSPACE_AUTHORITY_IDENTITY),
        },
        retained_pre_search_workspace: ResidentSearchSlice2TrustedReserveAuthorityV2 {
            expected_bytes: RETAINED_PRE_SEARCH_WORKSPACE_BYTES,
            expected_binding: trusted_binding(RETAINED_AUTHORITY_IDENTITY),
        },
        remaining_search_allocation_after_trim: ResidentSearchSlice2TrustedReserveAuthorityV2 {
            expected_bytes: REMAINING_SEARCH_ALLOCATION_AFTER_TRIM_BYTES,
            expected_binding: trusted_binding(REMAINING_AUTHORITY_IDENTITY),
        },
        same_context_free: ResidentSearchSlice2TrustedReserveAuthorityV2 {
            expected_bytes: SAME_CONTEXT_FREE_BYTES,
            expected_binding: trusted_binding(SAME_CONTEXT_FREE_AUTHORITY_IDENTITY),
        },
    };
    ResidentSearchSlice2TrustedReserveSealV2 {
        trusted_reserve,
        expected_calibration: valid_calibration(),
        sealed_full_workspace_receipt_identity: FULL_WORKSPACE_RECEIPT_IDENTITY,
        sealed_post_trim_receipt_identity: POST_TRIM_RECEIPT_IDENTITY,
    }
}

// BEGIN R6 TRUSTED SEAL INSPECTOR V2
fn assert_expected_authority_binding(
    binding: &ResidentSearchSlice2AuthorityBindingV2,
    authority_identity: u64,
) {
    assert_eq!(binding.device_uuid, DEVICE_UUID);
    assert_eq!(binding.primary_context_identity, PRIMARY_CONTEXT_IDENTITY);
    assert_eq!(binding.search_stream_identity, SEARCH_STREAM_IDENTITY);
    assert_eq!(binding.active_pool_identity, ACTIVE_POOL_IDENTITY);
    assert_eq!(binding.run_identity, RUN_IDENTITY);
    assert_eq!(
        binding.full_workspace_receipt_identity,
        FULL_WORKSPACE_RECEIPT_IDENTITY
    );
    assert_eq!(
        binding.post_trim_receipt_identity,
        POST_TRIM_RECEIPT_IDENTITY
    );
    assert_eq!(binding.authority_identity, authority_identity);
}

#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
fn assert_r6_trusted_reserve_seal_fixture_v2(seal: &ResidentSearchSlice2TrustedReserveSealV2) {
    assert_eq!(
        seal.trusted_reserve
            .allocator_context_headroom
            .expected_bytes,
        ALLOCATOR_CONTEXT_HEADROOM_BYTES
    );
    assert_expected_authority_binding(
        &seal
            .trusted_reserve
            .allocator_context_headroom
            .expected_binding,
        HEADROOM_AUTHORITY_IDENTITY,
    );
    assert_eq!(
        seal.trusted_reserve.full_workspace_authority.expected_bytes,
        FULL_WORKSPACE_AUTHORITY_BYTES
    );
    assert_expected_authority_binding(
        &seal
            .trusted_reserve
            .full_workspace_authority
            .expected_binding,
        FULL_WORKSPACE_AUTHORITY_IDENTITY,
    );
    assert_eq!(
        seal.trusted_reserve
            .retained_pre_search_workspace
            .expected_bytes,
        RETAINED_PRE_SEARCH_WORKSPACE_BYTES
    );
    assert_expected_authority_binding(
        &seal
            .trusted_reserve
            .retained_pre_search_workspace
            .expected_binding,
        RETAINED_AUTHORITY_IDENTITY,
    );
    assert_eq!(
        seal.trusted_reserve
            .remaining_search_allocation_after_trim
            .expected_bytes,
        REMAINING_SEARCH_ALLOCATION_AFTER_TRIM_BYTES
    );
    assert_expected_authority_binding(
        &seal
            .trusted_reserve
            .remaining_search_allocation_after_trim
            .expected_binding,
        REMAINING_AUTHORITY_IDENTITY,
    );
    assert_eq!(
        seal.trusted_reserve.same_context_free.expected_bytes,
        SAME_CONTEXT_FREE_BYTES
    );
    assert_expected_authority_binding(
        &seal.trusted_reserve.same_context_free.expected_binding,
        SAME_CONTEXT_FREE_AUTHORITY_IDENTITY,
    );
    assert_eq!(seal.expected_calibration.device_uuid, DEVICE_UUID);
    assert_eq!(
        seal.expected_calibration.primary_context_identity,
        PRIMARY_CONTEXT_IDENTITY
    );
    assert_eq!(
        seal.expected_calibration.search_stream_identity,
        SEARCH_STREAM_IDENTITY
    );
    assert_eq!(
        seal.expected_calibration.active_pool_identity,
        ACTIVE_POOL_IDENTITY
    );
    assert_eq!(
        seal.expected_calibration.cuda_build_identity,
        CUDA_BUILD_IDENTITY
    );
    assert_eq!(
        seal.expected_calibration.kernel_semantics_identity,
        KERNEL_SEMANTICS_IDENTITY
    );
    assert_eq!(
        seal.expected_calibration.binary64_math_identity,
        BINARY64_MATH_IDENTITY
    );
    assert_eq!(seal.expected_calibration.plan_identity, PLAN_IDENTITY);
    assert_eq!(seal.expected_calibration.run_identity, RUN_IDENTITY);
    assert_eq!(
        seal.sealed_full_workspace_receipt_identity,
        FULL_WORKSPACE_RECEIPT_IDENTITY
    );
    assert_eq!(
        seal.sealed_post_trim_receipt_identity,
        POST_TRIM_RECEIPT_IDENTITY
    );
}
// END R6 TRUSTED SEAL INSPECTOR V2

fn valid_request() -> ResidentSearchSlice2AdmissionRequestV2 {
    let layout = valid_slice2_layout();
    let generation_receipt = valid_generation_receipt();
    let scoring_archive_receipt = scoring_archive_receipt(layout);
    assert_eq!(
        scoring_archive_receipt.total_device_bytes,
        SCORING_ARCHIVE_TOTAL_BYTES
    );
    assert_eq!(
        generation_receipt
            .total_device_bytes
            .checked_add(scoring_archive_receipt.total_device_bytes)
            .expect("canonical requested-device sum must fit"),
        REQUESTED_DEVICE_SUM_BYTES
    );
    ResidentSearchSlice2AdmissionRequestV2 {
        population_count: POPULATION_COUNT,
        archive_capacity: ARCHIVE_CAPACITY,
        signature_word_count: SIGNATURE_WORD_COUNT,
        novelty_neighbor_count: NOVELTY_NEIGHBOR_COUNT,
        max_terms_per_gene: MAX_TERMS_PER_GENE,
        terminal_host_receipt_bytes: TERMINAL_HOST_RECEIPT_BYTES,
        terminal_host_alignment_bytes: TERMINAL_HOST_ALIGNMENT_BYTES,
        device_alignment_bytes: SLICE2_ALIGNMENT_BYTES,
        terminal_host_flags: CUDA_HOST_ALLOC_PORTABLE,
        archive_arena_present: true,
        archive_arena_bytes: archive_subreceipt_bytes(&layout),
        expected_slice2_layout: layout,
        generation_receipt,
        scoring_archive_receipt,
        observed_reserve: valid_observed_reserve(),
        calibration: valid_calibration(),
    }
}

fn requested_device_sum(request: &ResidentSearchSlice2AdmissionRequestV2) -> u64 {
    request
        .generation_receipt
        .total_device_bytes
        .checked_add(request.scoring_archive_receipt.total_device_bytes)
        .expect("fixture requested-device sum must fit")
}

fn expected_ledger(
    request: &ResidentSearchSlice2AdmissionRequestV2,
) -> Vec<ResidentSearchSlice2AllocationCallV2> {
    vec![
        ResidentSearchSlice2AllocationCallV2 {
            ordinal: 0,
            symbol: ResidentSearchSlice2AllocationSymbolV2::CudaHostAlloc,
            category: ResidentSearchSlice2AllocationCategoryV2::TerminalHostReceipt,
            requested_bytes: request.terminal_host_receipt_bytes,
            aligned_bytes: request.terminal_host_receipt_bytes,
            alignment_bytes: request.terminal_host_alignment_bytes,
            flags: request.terminal_host_flags,
            stream_identity: None,
            pool_identity: None,
        },
        ResidentSearchSlice2AllocationCallV2 {
            ordinal: 1,
            symbol: ResidentSearchSlice2AllocationSymbolV2::CudaMallocAsync,
            category: ResidentSearchSlice2AllocationCategoryV2::GenerationArena,
            requested_bytes: request.generation_receipt.total_device_bytes,
            aligned_bytes: request.generation_receipt.total_device_bytes,
            alignment_bytes: request.device_alignment_bytes,
            flags: 0,
            stream_identity: Some(request.calibration.search_stream_identity),
            pool_identity: Some(request.calibration.active_pool_identity),
        },
        ResidentSearchSlice2AllocationCallV2 {
            ordinal: 2,
            symbol: ResidentSearchSlice2AllocationSymbolV2::CudaMallocAsync,
            category: ResidentSearchSlice2AllocationCategoryV2::ScoringArchiveArena,
            requested_bytes: request.scoring_archive_receipt.total_device_bytes,
            aligned_bytes: request.scoring_archive_receipt.total_device_bytes,
            alignment_bytes: request.device_alignment_bytes,
            flags: 0,
            stream_identity: Some(request.calibration.search_stream_identity),
            pool_identity: Some(request.calibration.active_pool_identity),
        },
    ]
}

fn expected_chronology(
    ledger: &[ResidentSearchSlice2AllocationCallV2],
) -> Vec<ResidentSearchSlice2RecorderEventV2> {
    let mut chronology = vec![ResidentSearchSlice2RecorderEventV2::NativeCreate {
        phase_before: ResidentSearchSlice2RecorderPhaseV2::BeforeNativeCreate,
    }];
    chronology.extend(ledger.iter().copied().map(|call| {
        ResidentSearchSlice2RecorderEventV2::Allocation {
            phase_at_call: ResidentSearchSlice2RecorderPhaseV2::NativeCreateBegun,
            call,
        }
    }));
    chronology
}

fn assert_mutation_register_is_frozen() {
    assert_eq!(R6_MUTATION_NAMES.len(), 132);
    let mut sorted = R6_MUTATION_NAMES.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 132, "mutation names must remain unique");
    let _begin_native_create =
        <ResidentSearchSlice2AllocationRecorderV2 as ResidentSearchSlice2AllocationFacadeV2>::begin_native_create
            as fn(&mut ResidentSearchSlice2AllocationRecorderV2);
    let _cuda_host_alloc =
        <ResidentSearchSlice2AllocationRecorderV2 as ResidentSearchSlice2AllocationFacadeV2>::cuda_host_alloc
            as fn(
                &mut ResidentSearchSlice2AllocationRecorderV2,
                ResidentSearchSlice2HostAllocationArgsV2,
            );
    let _cuda_malloc_async =
        <ResidentSearchSlice2AllocationRecorderV2 as ResidentSearchSlice2AllocationFacadeV2>::cuda_malloc_async
            as fn(
                &mut ResidentSearchSlice2AllocationRecorderV2,
                ResidentSearchSlice2AsyncAllocationArgsV2,
            );
}

fn assert_zero_before_native_create(recorder: &ResidentSearchSlice2AllocationRecorderV2) {
    assert_eq!(
        recorder.phase(),
        ResidentSearchSlice2RecorderPhaseV2::BeforeNativeCreate
    );
    assert_eq!(recorder.native_create_count(), 0);
    assert_eq!(recorder.host_allocator_method_count(), 0);
    assert_eq!(recorder.async_allocator_method_count(), 0);
    assert_eq!(recorder.physical_allocator_count(), 0);
    assert_eq!(recorder.generation_arena_count(), 0);
    assert_eq!(recorder.scoring_archive_arena_count(), 0);
    assert_eq!(recorder.archive_only_arena_count(), 0);
    assert!(recorder.chronology().is_empty());
    assert!(recorder.observed().is_empty());
}

fn admit_with_pristine_seal(
    request: ResidentSearchSlice2AdmissionRequestV2,
    recorder: &mut ResidentSearchSlice2AllocationRecorderV2,
) -> Result<ResidentSearchSlice2AdmissionOwnerV2, ResidentSearchSlice2AdmissionErrorV2> {
    let seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
    assert_r6_trusted_reserve_seal_fixture_v2(&seal);
    admit_slice2_combined_fixture_v2(request, seal, recorder)
}

fn observed_layout_field(
    layout: &ResidentSearchSlice2AlignedLayoutV2,
    field: ResidentSearchSlice2AlignedFieldV2,
) -> u64 {
    match field {
        ResidentSearchSlice2AlignedFieldV2::ArchiveGeneScalars => layout.archive_gene_scalars,
        ResidentSearchSlice2AlignedFieldV2::ArchiveTermIndices => layout.archive_term_indices,
        ResidentSearchSlice2AlignedFieldV2::ArchiveTermWeights => layout.archive_term_weights,
        ResidentSearchSlice2AlignedFieldV2::ArchiveMetricRows => layout.archive_metric_rows,
        ResidentSearchSlice2AlignedFieldV2::ArchiveSignatures => layout.archive_signatures,
        ResidentSearchSlice2AlignedFieldV2::ArchiveHashes => layout.archive_hashes,
        ResidentSearchSlice2AlignedFieldV2::CurrentPopulationSignatures => {
            layout.current_population_signatures
        }
        ResidentSearchSlice2AlignedFieldV2::NoveltyScores => layout.novelty_scores,
        ResidentSearchSlice2AlignedFieldV2::ExactTopKKeys => layout.exact_top_k_keys,
        ResidentSearchSlice2AlignedFieldV2::AdmissionFlags => layout.admission_flags,
        ResidentSearchSlice2AlignedFieldV2::AdmissionOffsets => layout.admission_offsets,
        ResidentSearchSlice2AlignedFieldV2::ArchiveControlAndSeal => {
            layout.archive_control_and_seal
        }
    }
}

fn decrement_layout_field(
    layout: &mut ResidentSearchSlice2AlignedLayoutV2,
    field: ResidentSearchSlice2AlignedFieldV2,
) {
    let value = match field {
        ResidentSearchSlice2AlignedFieldV2::ArchiveGeneScalars => &mut layout.archive_gene_scalars,
        ResidentSearchSlice2AlignedFieldV2::ArchiveTermIndices => &mut layout.archive_term_indices,
        ResidentSearchSlice2AlignedFieldV2::ArchiveTermWeights => &mut layout.archive_term_weights,
        ResidentSearchSlice2AlignedFieldV2::ArchiveMetricRows => &mut layout.archive_metric_rows,
        ResidentSearchSlice2AlignedFieldV2::ArchiveSignatures => &mut layout.archive_signatures,
        ResidentSearchSlice2AlignedFieldV2::ArchiveHashes => &mut layout.archive_hashes,
        ResidentSearchSlice2AlignedFieldV2::CurrentPopulationSignatures => {
            &mut layout.current_population_signatures
        }
        ResidentSearchSlice2AlignedFieldV2::NoveltyScores => &mut layout.novelty_scores,
        ResidentSearchSlice2AlignedFieldV2::ExactTopKKeys => &mut layout.exact_top_k_keys,
        ResidentSearchSlice2AlignedFieldV2::AdmissionFlags => &mut layout.admission_flags,
        ResidentSearchSlice2AlignedFieldV2::AdmissionOffsets => &mut layout.admission_offsets,
        ResidentSearchSlice2AlignedFieldV2::ArchiveControlAndSeal => {
            &mut layout.archive_control_and_seal
        }
    };
    *value = value
        .checked_sub(1)
        .expect("every valid layout field is nonzero");
    layout.replacement_subtotal_bytes = slice2_layout_subtotal(layout);
}

fn observed_authority_mut(
    reserve: &mut ResidentSearchSlice2ObservedReserveSetV2,
    authority: ResidentSearchSlice2ReserveAuthorityKindV2,
) -> &mut ResidentSearchSlice2ObservedReserveAuthorityV2 {
    match authority {
        ResidentSearchSlice2ReserveAuthorityKindV2::AllocatorContextHeadroom => {
            &mut reserve.allocator_context_headroom
        }
        ResidentSearchSlice2ReserveAuthorityKindV2::FullWorkspaceAuthority => {
            &mut reserve.full_workspace_authority
        }
        ResidentSearchSlice2ReserveAuthorityKindV2::RetainedPreSearchWorkspace => {
            &mut reserve.retained_pre_search_workspace
        }
        ResidentSearchSlice2ReserveAuthorityKindV2::RemainingSearchAllocationAfterTrim => {
            &mut reserve.remaining_search_allocation_after_trim
        }
        ResidentSearchSlice2ReserveAuthorityKindV2::SameContextFree => {
            &mut reserve.same_context_free
        }
    }
}

fn trusted_authority_mut(
    seal: &mut ResidentSearchSlice2TrustedReserveSealV2,
    authority: ResidentSearchSlice2ReserveAuthorityKindV2,
) -> &mut ResidentSearchSlice2TrustedReserveAuthorityV2 {
    match authority {
        ResidentSearchSlice2ReserveAuthorityKindV2::AllocatorContextHeadroom => {
            &mut seal.trusted_reserve.allocator_context_headroom
        }
        ResidentSearchSlice2ReserveAuthorityKindV2::FullWorkspaceAuthority => {
            &mut seal.trusted_reserve.full_workspace_authority
        }
        ResidentSearchSlice2ReserveAuthorityKindV2::RetainedPreSearchWorkspace => {
            &mut seal.trusted_reserve.retained_pre_search_workspace
        }
        ResidentSearchSlice2ReserveAuthorityKindV2::RemainingSearchAllocationAfterTrim => {
            &mut seal.trusted_reserve.remaining_search_allocation_after_trim
        }
        ResidentSearchSlice2ReserveAuthorityKindV2::SameContextFree => {
            &mut seal.trusted_reserve.same_context_free
        }
    }
}

fn mutate_binding_axis(
    binding: &mut ResidentSearchSlice2AuthorityBindingV2,
    axis: ResidentSearchSlice2AuthorityBindingAxisV2,
    high_width: bool,
) {
    let mask = if high_width { 1_u64 << 63 } else { 1 };
    match axis {
        ResidentSearchSlice2AuthorityBindingAxisV2::DeviceUuid => {
            let index = if high_width { 15 } else { 0 };
            binding.device_uuid[index] ^= 0xff;
        }
        ResidentSearchSlice2AuthorityBindingAxisV2::PrimaryContext => {
            binding.primary_context_identity ^= mask;
        }
        ResidentSearchSlice2AuthorityBindingAxisV2::SearchStream => {
            binding.search_stream_identity ^= mask;
        }
        ResidentSearchSlice2AuthorityBindingAxisV2::ActivePool => {
            binding.active_pool_identity ^= mask;
        }
        ResidentSearchSlice2AuthorityBindingAxisV2::RunIdentity => {
            binding.run_identity ^= mask;
        }
        ResidentSearchSlice2AuthorityBindingAxisV2::FullWorkspaceReceiptIdentity => {
            binding.full_workspace_receipt_identity ^= mask;
        }
        ResidentSearchSlice2AuthorityBindingAxisV2::PostTrimReceiptIdentity => {
            binding.post_trim_receipt_identity ^= mask;
        }
        ResidentSearchSlice2AuthorityBindingAxisV2::AuthorityIdentity => {
            binding.authority_identity ^= mask;
        }
    }
}

fn assert_v8_source_topology_is_frozen() {
    fn identifier_count(source: &str, token: &str) -> usize {
        fn is_identifier_byte(byte: u8) -> bool {
            byte.is_ascii_alphanumeric() || byte == b'_'
        }

        let bytes = source.as_bytes();
        source
            .match_indices(token)
            .filter(|(index, _)| {
                let before_is_identifier = index
                    .checked_sub(1)
                    .and_then(|before| bytes.get(before))
                    .is_some_and(|byte| is_identifier_byte(*byte));
                let after_is_identifier = bytes
                    .get(index + token.len())
                    .is_some_and(|byte| is_identifier_byte(*byte));
                !before_is_identifier && !after_is_identifier
            })
            .count()
    }

    let shared = include_str!("resident_search_slice2_admission_v2.rs").replace("\r\n", "\n");
    let child = include_str!("resident_search_v2_tests.rs").replace("\r\n", "\n");
    let trusted_authority = "#[derive(Debug, PartialEq, Eq)]\npub(crate) struct ResidentSearchSlice2TrustedReserveAuthorityV2 {\n    expected_bytes: u64,\n    expected_binding: ResidentSearchSlice2AuthorityBindingV2,\n}";
    let trusted_set = "#[derive(Debug, PartialEq, Eq)]\npub(crate) struct ResidentSearchSlice2TrustedReserveSetV2 {\n    allocator_context_headroom: ResidentSearchSlice2TrustedReserveAuthorityV2,\n    full_workspace_authority: ResidentSearchSlice2TrustedReserveAuthorityV2,\n    retained_pre_search_workspace: ResidentSearchSlice2TrustedReserveAuthorityV2,\n    remaining_search_allocation_after_trim: ResidentSearchSlice2TrustedReserveAuthorityV2,\n    same_context_free: ResidentSearchSlice2TrustedReserveAuthorityV2,\n}";
    let trusted_seal = "pub(crate) struct ResidentSearchSlice2TrustedReserveSealV2 {\n    trusted_reserve: ResidentSearchSlice2TrustedReserveSetV2,\n    expected_calibration: ResidentSearchSlice2CalibrationBindingV2,\n    sealed_full_workspace_receipt_identity: u64,\n    sealed_post_trim_receipt_identity: u64,\n}";
    for block in [trusted_authority, trusted_set, trusted_seal] {
        assert_eq!(shared.matches(block).count(), 1);
    }
    for forbidden in [
        "impl ResidentSearchSlice2TrustedReserveAuthorityV2",
        "impl ResidentSearchSlice2TrustedReserveSetV2",
        "impl ResidentSearchSlice2TrustedReserveSealV2",
        "Clone for ResidentSearchSlice2Trusted",
        "Copy for ResidentSearchSlice2Trusted",
        "Default for ResidentSearchSlice2Trusted",
    ] {
        assert!(
            !shared.contains(forbidden),
            "forbidden trusted surface: {forbidden}"
        );
    }
    let trusted_type_names = [
        "ResidentSearchSlice2TrustedReserveAuthorityV2",
        "ResidentSearchSlice2TrustedReserveSetV2",
        "ResidentSearchSlice2TrustedReserveSealV2",
    ];
    let shared_flat = shared.split_whitespace().collect::<Vec<_>>().join(" ");
    for function_tail in shared_flat.split("fn ").skip(1) {
        let brace = function_tail.find('{').unwrap_or(function_tail.len());
        let semicolon = function_tail.find(';').unwrap_or(function_tail.len());
        let signature = &function_tail[..brace.min(semicolon)];
        if let Some((_, return_type)) = signature.split_once("->") {
            for trusted_type_name in trusted_type_names {
                assert!(
                    !return_type.contains(trusted_type_name),
                    "shared function returns trusted capability: {signature}"
                );
            }
        }
    }
    assert_eq!(
        shared
            .matches("_trusted_seal: ResidentSearchSlice2TrustedReserveSealV2,")
            .count(),
        1
    );
    let minter_signature = "#[cfg(all(test, feature = \"resident-search-slice2-host-contract\"))]\nfn mint_r6_trusted_reserve_seal_for_fixture_v2() -> ResidentSearchSlice2TrustedReserveSealV2 {";
    let inspector_signature = "#[cfg(all(test, feature = \"resident-search-slice2-host-contract\"))]\nfn assert_r6_trusted_reserve_seal_fixture_v2(seal: &ResidentSearchSlice2TrustedReserveSealV2) {";
    assert_eq!(child.matches(minter_signature).count(), 1);
    assert_eq!(child.matches(inspector_signature).count(), 1);
    let forbidden_generic_recorder_method = ["fn record_", "allocation("].concat();
    assert!(!child.contains(&forbidden_generic_recorder_method));
    for line in child.lines() {
        let trimmed = line.trim();
        assert!(
            !trimmed.starts_with("macro_rules!"),
            "trusted construction census forbids child macros"
        );
        for trusted_type_name in trusted_type_names {
            assert!(
                !(trimmed.starts_with("type ") && trimmed.contains(trusted_type_name)),
                "trusted construction census forbids child type aliases"
            );
            assert!(
                !(trimmed.starts_with("use ")
                    && trimmed.contains(trusted_type_name)
                    && trimmed.contains(" as ")),
                "trusted construction census forbids child use aliases"
            );
        }
    }
    assert_eq!(
        child
            .lines()
            .filter(|line| line.trim() == "ResidentSearchSlice2TrustedReserveSealV2 {")
            .count(),
        1
    );
    let inspector_begin = ["// BEGIN R6 TRUSTED ", "SEAL INSPECTOR V2"].concat();
    let inspector_end = ["// END R6 TRUSTED ", "SEAL INSPECTOR V2"].concat();
    assert_eq!(child.matches(&inspector_begin).count(), 1);
    assert_eq!(child.matches(&inspector_end).count(), 1);
    let inspector_begin_index = child
        .find(&inspector_begin)
        .expect("trusted seal inspector begin marker must exist");
    let inspector_end_index = child
        .find(&inspector_end)
        .expect("trusted seal inspector end marker must exist");
    assert!(inspector_end_index > inspector_begin_index);
    let inspector = &child[inspector_begin_index..inspector_end_index];
    let inspector_token_counts = [
        ("allocator_context_headroom", 2),
        ("full_workspace_authority", 2),
        ("retained_pre_search_workspace", 2),
        ("remaining_search_allocation_after_trim", 2),
        ("same_context_free", 2),
        ("expected_bytes", 5),
        ("expected_binding", 5),
        ("device_uuid", 2),
        ("primary_context_identity", 2),
        ("search_stream_identity", 2),
        ("active_pool_identity", 2),
        ("run_identity", 2),
        ("full_workspace_receipt_identity", 1),
        ("post_trim_receipt_identity", 1),
        ("authority_identity", 3),
        ("expected_calibration", 9),
        ("cuda_build_identity", 1),
        ("kernel_semantics_identity", 1),
        ("binary64_math_identity", 1),
        ("plan_identity", 1),
        ("sealed_full_workspace_receipt_identity", 1),
        ("sealed_post_trim_receipt_identity", 1),
        ("assert_expected_authority_binding", 6),
    ];
    for (token, expected_count) in inspector_token_counts {
        assert_eq!(
            identifier_count(inspector, token),
            expected_count,
            "trusted seal inspector token drifted: {token}"
        );
    }
}

#[test]
fn slice2_combined_admission_rejects_missing_or_zero_archive_arena_before_allocation() {
    assert_mutation_register_is_frozen();
    let mut missing = valid_request();
    missing.archive_arena_present = false;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(missing, &mut recorder);
    assert_eq!(
        actual.expect_err("missing archive authority must fail"),
        ResidentSearchSlice2AdmissionErrorV2::MissingArchiveArena
    );
    assert_zero_before_native_create(&recorder);

    let mut zero = valid_request();
    zero.archive_arena_bytes = 0;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(zero, &mut recorder);
    assert_eq!(
        actual.expect_err("zero archive bytes must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ZeroArchiveArenaBytes
    );
    assert_zero_before_native_create(&recorder);

    for axis in [
        ResidentSearchSlice2ShapeAxisV2::PopulationCount,
        ResidentSearchSlice2ShapeAxisV2::ArchiveCapacity,
        ResidentSearchSlice2ShapeAxisV2::SignatureWordCount,
        ResidentSearchSlice2ShapeAxisV2::NoveltyNeighborCount,
        ResidentSearchSlice2ShapeAxisV2::MaxTermsPerGene,
    ] {
        let mut request = valid_request();
        let (expected, observed) = match axis {
            ResidentSearchSlice2ShapeAxisV2::PopulationCount => {
                request.population_count += 1;
                (POPULATION_COUNT, request.population_count)
            }
            ResidentSearchSlice2ShapeAxisV2::ArchiveCapacity => {
                request.archive_capacity += 1;
                (ARCHIVE_CAPACITY, request.archive_capacity)
            }
            ResidentSearchSlice2ShapeAxisV2::SignatureWordCount => {
                request.signature_word_count += 1;
                (
                    u64::from(SIGNATURE_WORD_COUNT),
                    u64::from(request.signature_word_count),
                )
            }
            ResidentSearchSlice2ShapeAxisV2::NoveltyNeighborCount => {
                request.novelty_neighbor_count += 1;
                (
                    u64::from(NOVELTY_NEIGHBOR_COUNT),
                    u64::from(request.novelty_neighbor_count),
                )
            }
            ResidentSearchSlice2ShapeAxisV2::MaxTermsPerGene => {
                request.max_terms_per_gene += 1;
                (
                    u64::from(MAX_TERMS_PER_GENE),
                    u64::from(request.max_terms_per_gene),
                )
            }
        };
        let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
        let actual = admit_with_pristine_seal(request, &mut recorder);
        assert_eq!(
            actual.expect_err("shape drift must fail before authority mint"),
            ResidentSearchSlice2AdmissionErrorV2::ShapeMismatch {
                axis,
                expected,
                observed,
            }
        );
        assert_zero_before_native_create(&recorder);
    }
}

#[test]
fn slice2_combined_admission_rejects_each_aligned_layout_field_mismatch_before_allocation() {
    assert_mutation_register_is_frozen();
    let fields = [
        ResidentSearchSlice2AlignedFieldV2::ArchiveGeneScalars,
        ResidentSearchSlice2AlignedFieldV2::ArchiveTermIndices,
        ResidentSearchSlice2AlignedFieldV2::ArchiveTermWeights,
        ResidentSearchSlice2AlignedFieldV2::ArchiveMetricRows,
        ResidentSearchSlice2AlignedFieldV2::ArchiveSignatures,
        ResidentSearchSlice2AlignedFieldV2::ArchiveHashes,
        ResidentSearchSlice2AlignedFieldV2::CurrentPopulationSignatures,
        ResidentSearchSlice2AlignedFieldV2::NoveltyScores,
        ResidentSearchSlice2AlignedFieldV2::ExactTopKKeys,
        ResidentSearchSlice2AlignedFieldV2::AdmissionFlags,
        ResidentSearchSlice2AlignedFieldV2::AdmissionOffsets,
        ResidentSearchSlice2AlignedFieldV2::ArchiveControlAndSeal,
    ];
    assert_eq!(fields.len(), 12);
    let mut control_count = 0;
    for field in fields {
        let mut request = valid_request();
        let expected_aligned_bytes = observed_layout_field(&request.expected_slice2_layout, field);
        let mut mutated_layout = request.scoring_archive_receipt.layout;
        decrement_layout_field(&mut mutated_layout, field);
        let observed_aligned_bytes = observed_layout_field(&mutated_layout, field);
        request.archive_arena_bytes = archive_subreceipt_bytes(&mutated_layout);
        request.scoring_archive_receipt = scoring_archive_receipt(mutated_layout);
        let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
        let actual = admit_with_pristine_seal(request, &mut recorder);
        assert_eq!(
            actual.expect_err("every named aligned-field mutation must fail"),
            ResidentSearchSlice2AdmissionErrorV2::AlignedLayoutFieldMismatch {
                field,
                expected_aligned_bytes,
                observed_aligned_bytes,
            }
        );
        assert_zero_before_native_create(&recorder);
        control_count += 1;
    }
    let mut request = valid_request();
    let expected_total_bytes = request
        .scoring_archive_receipt
        .layout
        .replacement_subtotal_bytes;
    request
        .scoring_archive_receipt
        .layout
        .replacement_subtotal_bytes = expected_total_bytes + 1;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("replacement subtotal-only drift must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReceiptTotalMismatch {
            axis: ResidentSearchSlice2ReceiptTotalAxisV2::ReplacementSubtotal,
            expected_total_bytes,
            observed_total_bytes: expected_total_bytes + 1,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    request.expected_slice2_layout.archive_gene_scalars = u64::MAX;
    request.expected_slice2_layout.archive_term_indices = 1;
    request.expected_slice2_layout.replacement_subtotal_bytes = 7;
    request.scoring_archive_receipt.layout = request.expected_slice2_layout;
    request.scoring_archive_receipt.total_device_bytes = 9;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("replacement checked-add overflow must precede total mismatch"),
        ResidentSearchSlice2AdmissionErrorV2::ReceiptArithmeticOverflow {
            operation: ResidentSearchSlice2ReceiptArithmeticV2::ReplacementSubtotalAdd,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    assert_eq!(control_count, 14);
}

#[test]
fn slice2_combined_admission_rejects_insufficient_reserve_before_allocation() {
    assert_mutation_register_is_frozen();
    assert_v8_source_topology_is_frozen();
    let mut control_count = 0;
    let mut request = valid_request();
    let expected_total_bytes = generation_component_sum(&request.generation_receipt);
    request.generation_receipt.total_device_bytes = expected_total_bytes + 1;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("generation total-only drift must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReceiptTotalMismatch {
            axis: ResidentSearchSlice2ReceiptTotalAxisV2::GenerationReceiptTotal,
            expected_total_bytes,
            observed_total_bytes: expected_total_bytes + 1,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    request.generation_receipt.logical_gene_scalar_bytes = u64::MAX;
    request.generation_receipt.logical_gene_index_bytes = 1;
    request.generation_receipt.total_device_bytes = 0;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("generation checked-add overflow must precede total mismatch"),
        ResidentSearchSlice2AdmissionErrorV2::ReceiptArithmeticOverflow {
            operation: ResidentSearchSlice2ReceiptArithmeticV2::GenerationReceiptTotalAdd,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    assert_eq!(control_count, 2);

    let mut request = valid_request();
    let expected_total_bytes = request.scoring_archive_receipt.total_device_bytes;
    request.scoring_archive_receipt.total_device_bytes = expected_total_bytes + 1;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("scoring/archive total-only drift must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReceiptTotalMismatch {
            axis: ResidentSearchSlice2ReceiptTotalAxisV2::ScoringArchiveReceiptTotal,
            expected_total_bytes,
            observed_total_bytes: expected_total_bytes + 1,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    request.scoring_archive_receipt.fitness_score_bytes = u64::MAX;
    request.scoring_archive_receipt.decision_key_bytes = 1;
    request.scoring_archive_receipt.total_device_bytes = 0;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("scoring/archive checked-add overflow must precede total mismatch"),
        ResidentSearchSlice2AdmissionErrorV2::ReceiptArithmeticOverflow {
            operation: ResidentSearchSlice2ReceiptArithmeticV2::ScoringArchiveReceiptTotalAdd,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    assert_eq!(control_count, 4);

    let authorities = [
        ResidentSearchSlice2ReserveAuthorityKindV2::AllocatorContextHeadroom,
        ResidentSearchSlice2ReserveAuthorityKindV2::FullWorkspaceAuthority,
        ResidentSearchSlice2ReserveAuthorityKindV2::RetainedPreSearchWorkspace,
        ResidentSearchSlice2ReserveAuthorityKindV2::RemainingSearchAllocationAfterTrim,
        ResidentSearchSlice2ReserveAuthorityKindV2::SameContextFree,
    ];
    assert_eq!(authorities.len(), 5);
    for authority in authorities {
        let mut request = valid_request();
        let observed = observed_authority_mut(&mut request.observed_reserve, authority);
        let expected_bytes = observed.bytes;
        observed.bytes = expected_bytes + 1;
        let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
        let actual = admit_with_pristine_seal(request, &mut recorder);
        assert_eq!(
            actual.expect_err("every authority byte count is independently trusted"),
            ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityBytesMismatch {
                authority,
                expected_bytes,
                observed_bytes: expected_bytes + 1,
            }
        );
        assert_zero_before_native_create(&recorder);
        control_count += 1;
    }
    assert_eq!(control_count, 9);
    let binding_axes = [
        ResidentSearchSlice2AuthorityBindingAxisV2::DeviceUuid,
        ResidentSearchSlice2AuthorityBindingAxisV2::PrimaryContext,
        ResidentSearchSlice2AuthorityBindingAxisV2::SearchStream,
        ResidentSearchSlice2AuthorityBindingAxisV2::ActivePool,
        ResidentSearchSlice2AuthorityBindingAxisV2::RunIdentity,
        ResidentSearchSlice2AuthorityBindingAxisV2::FullWorkspaceReceiptIdentity,
        ResidentSearchSlice2AuthorityBindingAxisV2::PostTrimReceiptIdentity,
        ResidentSearchSlice2AuthorityBindingAxisV2::AuthorityIdentity,
    ];
    assert_eq!(binding_axes.len(), 8);
    let mut binding_width_controls = 0;
    for authority in authorities {
        for axis in binding_axes {
            for high_width in [false, true] {
                let mut request = valid_request();
                let observed = observed_authority_mut(&mut request.observed_reserve, authority);
                mutate_binding_axis(&mut observed.binding, axis, high_width);
                let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
                let actual = admit_with_pristine_seal(request, &mut recorder);
                assert_eq!(
                    actual.expect_err("every full-width binding axis must be enforced"),
                    ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityBindingMismatch {
                        authority,
                        axis,
                    }
                );
                assert_zero_before_native_create(&recorder);
                binding_width_controls += 1;
                control_count += 1;
            }
        }
    }
    assert_eq!(binding_width_controls, 80);
    assert_eq!(control_count, 89);

    let original_four = [
        ResidentSearchSlice2ReserveAuthorityKindV2::AllocatorContextHeadroom,
        ResidentSearchSlice2ReserveAuthorityKindV2::FullWorkspaceAuthority,
        ResidentSearchSlice2ReserveAuthorityKindV2::RetainedPreSearchWorkspace,
        ResidentSearchSlice2ReserveAuthorityKindV2::RemainingSearchAllocationAfterTrim,
    ];
    let mut request = valid_request();
    let mut seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
    assert_r6_trusted_reserve_seal_fixture_v2(&seal);
    for authority in original_four {
        observed_authority_mut(&mut request.observed_reserve, authority)
            .binding
            .authority_identity = PAIR_ALIAS_IDENTITY_V2;
        trusted_authority_mut(&mut seal, authority)
            .expected_binding
            .authority_identity = PAIR_ALIAS_IDENTITY_V2;
    }
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_slice2_combined_fixture_v2(request, seal, &mut recorder);
    assert_eq!(
        actual.expect_err("four-way reserve authority aliasing must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityRelationMismatch {
            relation: ResidentSearchSlice2ReserveRelationV2::FourReserveAuthorityIdentitiesDistinct,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    for canonical in [
        HEADROOM_AUTHORITY_IDENTITY,
        FULL_WORKSPACE_AUTHORITY_IDENTITY,
        RETAINED_AUTHORITY_IDENTITY,
        REMAINING_AUTHORITY_IDENTITY,
        SAME_CONTEXT_FREE_AUTHORITY_IDENTITY,
    ] {
        assert_ne!(PAIR_ALIAS_IDENTITY_V2, canonical);
    }
    let pair_cases = [
        (original_four[0], original_four[1]),
        (original_four[0], original_four[2]),
        (original_four[0], original_four[3]),
        (original_four[1], original_four[2]),
        (original_four[1], original_four[3]),
        (original_four[2], original_four[3]),
    ];
    assert_eq!(pair_cases.len(), 6);
    for (left, right) in pair_cases {
        let mut request = valid_request();
        let mut seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
        assert_r6_trusted_reserve_seal_fixture_v2(&seal);
        for authority in [left, right] {
            observed_authority_mut(&mut request.observed_reserve, authority)
                .binding
                .authority_identity = PAIR_ALIAS_IDENTITY_V2;
            trusted_authority_mut(&mut seal, authority)
                .expected_binding
                .authority_identity = PAIR_ALIAS_IDENTITY_V2;
        }
        let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
        let actual = admit_slice2_combined_fixture_v2(request, seal, &mut recorder);
        assert_eq!(
            actual.expect_err("every pair-only reserve identity alias must fail"),
            ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityRelationMismatch {
                relation:
                    ResidentSearchSlice2ReserveRelationV2::FourReserveAuthorityIdentitiesDistinct,
            }
        );
        assert_zero_before_native_create(&recorder);
        control_count += 1;
    }

    let mut request = valid_request();
    request.observed_reserve.retained_pre_search_workspace.bytes += 1;
    request.observed_reserve.full_workspace_authority.bytes += 1;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("coordinated retained/full substitution must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityBytesMismatch {
            authority: ResidentSearchSlice2ReserveAuthorityKindV2::FullWorkspaceAuthority,
            expected_bytes: FULL_WORKSPACE_AUTHORITY_BYTES,
            observed_bytes: FULL_WORKSPACE_AUTHORITY_BYTES + 1,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    request.observed_reserve.allocator_context_headroom.bytes += 1;
    request.observed_reserve.same_context_free.bytes += 1;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("coordinated headroom/free substitution must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityBytesMismatch {
            authority: ResidentSearchSlice2ReserveAuthorityKindV2::AllocatorContextHeadroom,
            expected_bytes: ALLOCATOR_CONTEXT_HEADROOM_BYTES,
            observed_bytes: ALLOCATOR_CONTEXT_HEADROOM_BYTES + 1,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    for authority in authorities {
        let observed = observed_authority_mut(&mut request.observed_reserve, authority);
        for axis in binding_axes {
            mutate_binding_axis(&mut observed.binding, axis, false);
        }
    }
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("coordinated observed binding substitution must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityBindingMismatch {
            authority: ResidentSearchSlice2ReserveAuthorityKindV2::AllocatorContextHeadroom,
            axis: ResidentSearchSlice2AuthorityBindingAxisV2::DeviceUuid,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    let mut plain_expected_looking_values = valid_observed_reserve();
    for reserve in [
        &mut request.observed_reserve,
        &mut plain_expected_looking_values,
    ] {
        for authority in authorities {
            let observed = observed_authority_mut(reserve, authority);
            observed.bytes += match authority {
                ResidentSearchSlice2ReserveAuthorityKindV2::FullWorkspaceAuthority => 4,
                _ => 2,
            };
            for axis in binding_axes {
                mutate_binding_axis(&mut observed.binding, axis, true);
            }
        }
    }
    assert_eq!(request.observed_reserve, plain_expected_looking_values);
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("plain expected-looking values cannot mint the trusted seal"),
        ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityBytesMismatch {
            authority: ResidentSearchSlice2ReserveAuthorityKindV2::AllocatorContextHeadroom,
            expected_bytes: ALLOCATOR_CONTEXT_HEADROOM_BYTES,
            observed_bytes: ALLOCATOR_CONTEXT_HEADROOM_BYTES + 2,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    assert_eq!(control_count, 100);

    let mut request = valid_request();
    let mut seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
    assert_r6_trusted_reserve_seal_fixture_v2(&seal);
    request.observed_reserve.full_workspace_authority.bytes = 91_127_041;
    seal.trusted_reserve.full_workspace_authority.expected_bytes = 91_127_041;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_slice2_combined_fixture_v2(request, seal, &mut recorder);
    assert_eq!(
        actual.expect_err("workspace partition mismatch must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityRelationMismatch {
            relation:
                ResidentSearchSlice2ReserveRelationV2::RetainedPlusRemainingEqualsFullWorkspace,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    let mut seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
    assert_r6_trusted_reserve_seal_fixture_v2(&seal);
    request.observed_reserve.retained_pre_search_workspace.bytes = u64::MAX;
    request
        .observed_reserve
        .remaining_search_allocation_after_trim
        .bytes = u64::MAX;
    seal.trusted_reserve
        .retained_pre_search_workspace
        .expected_bytes = u64::MAX;
    seal.trusted_reserve
        .remaining_search_allocation_after_trim
        .expected_bytes = u64::MAX;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_slice2_combined_fixture_v2(request, seal, &mut recorder);
    assert_eq!(
        actual.expect_err("workspace checked-add overflow must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReserveArithmeticOverflow {
            operation: ResidentSearchSlice2ReserveArithmeticV2::WorkspacePartitionAdd,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    let scoring_total = request.scoring_archive_receipt.total_device_bytes;
    let generation_total = u64::MAX - scoring_total + 1;
    request.generation_receipt = ResidentSearchSlice2GenerationReceiptV2 {
        logical_gene_scalar_bytes: generation_total,
        logical_gene_index_bytes: 0,
        logical_gene_weight_bytes: 0,
        offspring_bytes: 0,
        metric_row_bytes: 0,
        rank_key_bytes: 0,
        selection_bytes: 0,
        dedup_hash_bytes: 0,
        cub_scratch_bytes: 0,
        retained_evaluation_workspace_bytes: 0,
        terminal_device_receipt_bytes: 0,
        total_device_bytes: generation_total,
    };
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_with_pristine_seal(request, &mut recorder);
    assert_eq!(
        actual.expect_err("requested-device checked-add overflow must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReserveArithmeticOverflow {
            operation: ResidentSearchSlice2ReserveArithmeticV2::RequestedDeviceSumAdd,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    let mut seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
    assert_r6_trusted_reserve_seal_fixture_v2(&seal);
    request.observed_reserve.same_context_free.bytes = 0;
    request.observed_reserve.allocator_context_headroom.bytes = 1;
    seal.trusted_reserve.same_context_free.expected_bytes = 0;
    seal.trusted_reserve
        .allocator_context_headroom
        .expected_bytes = 1;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_slice2_combined_fixture_v2(request, seal, &mut recorder);
    assert_eq!(
        actual.expect_err("same-context checked subtraction underflow must fail"),
        ResidentSearchSlice2AdmissionErrorV2::ReserveArithmeticOverflow {
            operation: ResidentSearchSlice2ReserveArithmeticV2::SameContextFreeMinusHeadroom,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    assert_eq!(control_count, 104);

    let request = valid_request();
    assert_eq!(requested_device_sum(&request), REQUESTED_DEVICE_SUM_BYTES);
    let expected = expected_ledger(&request);
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let _owner = admit_with_pristine_seal(request, &mut recorder)
        .expect("the exact-fit control must admit without unlabelled slack");
    assert_eq!(recorder.observed(), expected.as_slice());
    control_count += 1;
    let mut request = valid_request();
    let mut seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
    assert_r6_trusted_reserve_seal_fixture_v2(&seal);
    request
        .observed_reserve
        .remaining_search_allocation_after_trim
        .bytes = 24_018_175;
    request.observed_reserve.full_workspace_authority.bytes = 91_127_039;
    seal.trusted_reserve
        .remaining_search_allocation_after_trim
        .expected_bytes = 24_018_175;
    seal.trusted_reserve.full_workspace_authority.expected_bytes = 91_127_039;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_slice2_combined_fixture_v2(request, seal, &mut recorder);
    assert_eq!(
        actual.expect_err("remaining budget one byte short must fail"),
        ResidentSearchSlice2AdmissionErrorV2::InsufficientAllocationBudget {
            axis: ResidentSearchSlice2AllocationBudgetAxisV2::RemainingSearchAllocationAfterTrim,
            required_bytes: REQUESTED_DEVICE_SUM_BYTES,
            available_bytes: 24_018_175,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    let mut request = valid_request();
    let mut seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
    assert_r6_trusted_reserve_seal_fixture_v2(&seal);
    request.observed_reserve.same_context_free.bytes = 32_406_783;
    seal.trusted_reserve.same_context_free.expected_bytes = 32_406_783;
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let actual = admit_slice2_combined_fixture_v2(request, seal, &mut recorder);
    assert_eq!(
        actual.expect_err("same-context budget one byte short must fail"),
        ResidentSearchSlice2AdmissionErrorV2::InsufficientAllocationBudget {
            axis: ResidentSearchSlice2AllocationBudgetAxisV2::SameContextFreeAfterAllocatorHeadroom,
            required_bytes: REQUESTED_DEVICE_SUM_BYTES,
            available_bytes: REQUESTED_DEVICE_SUM_BYTES - 1,
        }
    );
    assert_zero_before_native_create(&recorder);
    control_count += 1;
    assert_eq!(control_count, 107);
}

#[test]
fn slice2_combined_admission_rejects_foreign_calibration_before_allocation() {
    assert_mutation_register_is_frozen();
    let axes = [
        ResidentSearchSlice2CalibrationAxisV2::DeviceUuid,
        ResidentSearchSlice2CalibrationAxisV2::PrimaryContext,
        ResidentSearchSlice2CalibrationAxisV2::SearchStream,
        ResidentSearchSlice2CalibrationAxisV2::ActivePool,
        ResidentSearchSlice2CalibrationAxisV2::CudaBuildIdentity,
        ResidentSearchSlice2CalibrationAxisV2::KernelSemanticsIdentity,
        ResidentSearchSlice2CalibrationAxisV2::Binary64MathIdentity,
        ResidentSearchSlice2CalibrationAxisV2::PlanIdentity,
        ResidentSearchSlice2CalibrationAxisV2::RunIdentity,
    ];
    assert_eq!(axes.len(), 9);
    let mut mutation_count = 0;
    for axis in axes {
        for high_width in [false, true] {
            let mut request = valid_request();
            let mask = if high_width { 1_u64 << 63 } else { 1 };
            match axis {
                ResidentSearchSlice2CalibrationAxisV2::DeviceUuid => {
                    let index = if high_width { 15 } else { 0 };
                    request.calibration.device_uuid[index] ^= 0xff;
                }
                ResidentSearchSlice2CalibrationAxisV2::PrimaryContext => {
                    request.calibration.primary_context_identity ^= mask;
                }
                ResidentSearchSlice2CalibrationAxisV2::SearchStream => {
                    request.calibration.search_stream_identity ^= mask;
                }
                ResidentSearchSlice2CalibrationAxisV2::ActivePool => {
                    request.calibration.active_pool_identity ^= mask;
                }
                ResidentSearchSlice2CalibrationAxisV2::CudaBuildIdentity => {
                    request.calibration.cuda_build_identity ^= mask;
                }
                ResidentSearchSlice2CalibrationAxisV2::KernelSemanticsIdentity => {
                    request.calibration.kernel_semantics_identity ^= mask;
                }
                ResidentSearchSlice2CalibrationAxisV2::Binary64MathIdentity => {
                    request.calibration.binary64_math_identity ^= mask;
                }
                ResidentSearchSlice2CalibrationAxisV2::PlanIdentity => {
                    request.calibration.plan_identity ^= mask;
                }
                ResidentSearchSlice2CalibrationAxisV2::RunIdentity => {
                    request.calibration.run_identity ^= mask;
                }
            }
            let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
            let actual = admit_with_pristine_seal(request, &mut recorder);
            assert_eq!(
                actual.expect_err("every independent calibration axis must be bound"),
                ResidentSearchSlice2AdmissionErrorV2::ForeignCalibration { axis }
            );
            assert_zero_before_native_create(&recorder);
            mutation_count += 1;
        }
    }
    assert_eq!(mutation_count, 18);
}

fn host_args(
    category: ResidentSearchSlice2AllocationCategoryV2,
) -> ResidentSearchSlice2HostAllocationArgsV2 {
    ResidentSearchSlice2HostAllocationArgsV2 {
        ordinal: 7,
        category,
        requested_bytes: 512,
        aligned_bytes: 512,
        alignment_bytes: SLICE2_ALIGNMENT_BYTES,
        flags: CUDA_HOST_ALLOC_PORTABLE,
    }
}

fn async_args(
    category: ResidentSearchSlice2AllocationCategoryV2,
) -> ResidentSearchSlice2AsyncAllocationArgsV2 {
    ResidentSearchSlice2AsyncAllocationArgsV2 {
        ordinal: 8,
        category,
        requested_bytes: 768,
        aligned_bytes: 768,
        alignment_bytes: SLICE2_ALIGNMENT_BYTES,
        flags: 0,
        stream_identity: SEARCH_STREAM_IDENTITY,
        pool_identity: ACTIVE_POOL_IDENTITY,
    }
}

#[test]
fn slice2_valid_combined_admission_executes_declared_ledger_once_and_later_generations_allocate_nothing()
 {
    assert_mutation_register_is_frozen();
    let mut direct_control_count = 0;
    let mut async_terminal = ResidentSearchSlice2AllocationRecorderV2::default();
    async_terminal.cuda_malloc_async(async_args(
        ResidentSearchSlice2AllocationCategoryV2::TerminalHostReceipt,
    ));
    assert_eq!(async_terminal.async_allocator_method_count(), 1);
    assert_eq!(async_terminal.host_allocator_method_count(), 0);
    assert_eq!(
        async_terminal.observed()[0].symbol,
        ResidentSearchSlice2AllocationSymbolV2::CudaMallocAsync
    );
    direct_control_count += 1;
    let mut host_generation = ResidentSearchSlice2AllocationRecorderV2::default();
    host_generation.cuda_host_alloc(host_args(
        ResidentSearchSlice2AllocationCategoryV2::GenerationArena,
    ));
    assert_eq!(host_generation.host_allocator_method_count(), 1);
    assert_eq!(host_generation.async_allocator_method_count(), 0);
    assert_eq!(host_generation.generation_arena_count(), 1);
    assert_eq!(
        host_generation.observed()[0].symbol,
        ResidentSearchSlice2AllocationSymbolV2::CudaHostAlloc
    );
    direct_control_count += 1;
    let mut host_scoring = ResidentSearchSlice2AllocationRecorderV2::default();
    host_scoring.cuda_host_alloc(host_args(
        ResidentSearchSlice2AllocationCategoryV2::ScoringArchiveArena,
    ));
    assert_eq!(host_scoring.host_allocator_method_count(), 1);
    assert_eq!(host_scoring.scoring_archive_arena_count(), 1);
    assert_eq!(
        host_scoring.observed()[0].symbol,
        ResidentSearchSlice2AllocationSymbolV2::CudaHostAlloc
    );
    direct_control_count += 1;
    let mut direct_native_create = ResidentSearchSlice2AllocationRecorderV2::default();
    direct_native_create.begin_native_create();
    assert_eq!(direct_native_create.native_create_count(), 1);
    assert_eq!(
        direct_native_create.phase(),
        ResidentSearchSlice2RecorderPhaseV2::NativeCreateBegun
    );
    assert_eq!(
        direct_native_create.chronology(),
        &[ResidentSearchSlice2RecorderEventV2::NativeCreate {
            phase_before: ResidentSearchSlice2RecorderPhaseV2::BeforeNativeCreate,
        }]
    );
    assert!(direct_native_create.observed().is_empty());
    direct_control_count += 1;
    let mut allocation_before_create = ResidentSearchSlice2AllocationRecorderV2::default();
    allocation_before_create.cuda_malloc_async(async_args(
        ResidentSearchSlice2AllocationCategoryV2::ArchiveOnlyArena,
    ));
    assert_eq!(
        allocation_before_create.chronology()[0],
        ResidentSearchSlice2RecorderEventV2::Allocation {
            phase_at_call: ResidentSearchSlice2RecorderPhaseV2::BeforeNativeCreate,
            call: allocation_before_create.observed()[0],
        }
    );
    direct_control_count += 1;
    assert_eq!(direct_control_count, 5);

    let request = valid_request();
    assert_eq!(request.population_count, POPULATION_COUNT);
    assert_eq!(request.archive_capacity, ARCHIVE_CAPACITY);
    assert_eq!(request.signature_word_count, SIGNATURE_WORD_COUNT);
    assert_eq!(request.novelty_neighbor_count, NOVELTY_NEIGHBOR_COUNT);
    assert_eq!(request.max_terms_per_gene, MAX_TERMS_PER_GENE);
    assert_eq!(
        request
            .scoring_archive_receipt
            .layout
            .replacement_subtotal_bytes,
        REPLACEMENT_SUBTOTAL_BYTES
    );

    let pristine_seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
    assert_r6_trusted_reserve_seal_fixture_v2(&pristine_seal);
    let validated = validate_and_seal_slice2_combined_v2(request, pristine_seal)
        .expect("valid admission must mint runtime authority");
    let (_, _, _, runtime_authority) = validated.into_parts_v2();
    assert_eq!(runtime_authority.calibration, valid_calibration());
    assert_eq!(runtime_authority.observed_reserve, valid_observed_reserve());
    assert_eq!(
        runtime_authority.sealed_full_workspace_receipt_identity,
        FULL_WORKSPACE_RECEIPT_IDENTITY
    );
    assert_eq!(
        runtime_authority.sealed_post_trim_receipt_identity,
        POST_TRIM_RECEIPT_IDENTITY
    );
    assert_eq!(runtime_authority.population_count, POPULATION_COUNT);
    assert_eq!(runtime_authority.archive_capacity, ARCHIVE_CAPACITY);
    assert_eq!(runtime_authority.signature_word_count, SIGNATURE_WORD_COUNT);
    assert_eq!(
        runtime_authority.novelty_neighbor_count,
        NOVELTY_NEIGHBOR_COUNT
    );
    assert_eq!(runtime_authority.max_terms_per_gene, MAX_TERMS_PER_GENE);
    assert_eq!(
        runtime_authority
            .scoring_archive_layout
            .test_archive_gene_scalars_v2(),
        (69_120, 3_600_128)
    );
    assert_eq!(
        runtime_authority
            .scoring_archive_layout
            .test_total_device_bytes_v2(),
        SCORING_ARCHIVE_TOTAL_BYTES
    );

    let mut dynamic_request = valid_request();
    let dynamic_delta = 65_536;
    dynamic_request.scoring_archive_receipt.cub_scratch_bytes += dynamic_delta;
    dynamic_request.scoring_archive_receipt.total_device_bytes += dynamic_delta;
    dynamic_request
        .observed_reserve
        .remaining_search_allocation_after_trim
        .bytes += dynamic_delta;
    dynamic_request
        .observed_reserve
        .full_workspace_authority
        .bytes += dynamic_delta;
    dynamic_request.observed_reserve.same_context_free.bytes += dynamic_delta;
    let mut dynamic_seal = mint_r6_trusted_reserve_seal_for_fixture_v2();
    assert_r6_trusted_reserve_seal_fixture_v2(&dynamic_seal);
    dynamic_seal
        .trusted_reserve
        .remaining_search_allocation_after_trim
        .expected_bytes += dynamic_delta;
    dynamic_seal
        .trusted_reserve
        .full_workspace_authority
        .expected_bytes += dynamic_delta;
    dynamic_seal
        .trusted_reserve
        .same_context_free
        .expected_bytes += dynamic_delta;
    let dynamic_validated = validate_and_seal_slice2_combined_v2(dynamic_request, dynamic_seal)
        .expect("runtime CUB size must remain dynamic");
    let (_, _, _, dynamic_authority) = dynamic_validated.into_parts_v2();
    assert_eq!(dynamic_authority.calibration, valid_calibration());
    assert_eq!(
        dynamic_authority
            .scoring_archive_layout
            .test_archive_gene_scalars_v2(),
        (134_656, 3_600_128)
    );
    assert_eq!(
        dynamic_authority
            .scoring_archive_layout
            .test_total_device_bytes_v2(),
        23_842_304
    );

    let request = valid_request();
    let expected = expected_ledger(&request);
    let expected_events = expected_chronology(&expected);
    let mut recorder = ResidentSearchSlice2AllocationRecorderV2::default();
    let owner = admit_with_pristine_seal(request, &mut recorder)
        .expect("valid combined admission must return the move-only owner");
    assert_eq!(
        recorder.phase(),
        ResidentSearchSlice2RecorderPhaseV2::AllocationsComplete
    );
    assert_eq!(recorder.native_create_count(), 1);
    assert_eq!(recorder.host_allocator_method_count(), 1);
    assert_eq!(recorder.async_allocator_method_count(), 2);
    assert_eq!(recorder.physical_allocator_count(), 3);
    assert_eq!(recorder.generation_arena_count(), 1);
    assert_eq!(recorder.scoring_archive_arena_count(), 1);
    assert_eq!(recorder.archive_only_arena_count(), 0);
    assert_eq!(recorder.chronology().len(), 4);
    assert_eq!(recorder.chronology(), expected_events.as_slice());
    assert_eq!(recorder.observed().len(), 3);
    assert_eq!(recorder.observed(), expected.as_slice());
    let forbidden_category = ResidentSearchSlice2AllocationCategoryV2::ArchiveOnlyArena;
    assert!(
        recorder
            .observed()
            .iter()
            .all(|entry| entry.category != forbidden_category)
    );
    let admission_snapshot = recorder.snapshot();
    let owner = owner
        .queue_generation_v2(1, &mut recorder)
        .expect("generation one queue must retain the owner");
    assert_eq!(recorder, admission_snapshot);
    let owner = owner
        .queue_generation_v2(2, &mut recorder)
        .expect("generation two queue must retain the owner");
    assert_eq!(recorder, admission_snapshot);
    let _owner = owner
        .queue_generation_v2(3, &mut recorder)
        .expect("generation three queue must retain the owner");
    assert_eq!(recorder, admission_snapshot);
}
