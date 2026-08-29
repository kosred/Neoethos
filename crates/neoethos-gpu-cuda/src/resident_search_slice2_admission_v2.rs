#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AlignedFieldV2 {
    ArchiveGeneScalars,
    ArchiveTermIndices,
    ArchiveTermWeights,
    ArchiveMetricRows,
    ArchiveSignatures,
    ArchiveHashes,
    CurrentPopulationSignatures,
    NoveltyScores,
    ExactTopKKeys,
    AdmissionFlags,
    AdmissionOffsets,
    ArchiveControlAndSeal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReceiptTotalAxisV2 {
    ReplacementSubtotal,
    GenerationReceiptTotal,
    ScoringArchiveReceiptTotal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReceiptArithmeticV2 {
    ReplacementSubtotalAdd,
    GenerationReceiptTotalAdd,
    ScoringArchiveReceiptTotalAdd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReserveAuthorityKindV2 {
    AllocatorContextHeadroom,
    FullWorkspaceAuthority,
    RetainedPreSearchWorkspace,
    RemainingSearchAllocationAfterTrim,
    SameContextFree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AuthorityBindingAxisV2 {
    DeviceUuid,
    PrimaryContext,
    SearchStream,
    ActivePool,
    RunIdentity,
    FullWorkspaceReceiptIdentity,
    PostTrimReceiptIdentity,
    AuthorityIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReserveRelationV2 {
    FourReserveAuthorityIdentitiesDistinct,
    RetainedPlusRemainingEqualsFullWorkspace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReserveArithmeticV2 {
    WorkspacePartitionAdd,
    RequestedDeviceSumAdd,
    SameContextFreeMinusHeadroom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AllocationBudgetAxisV2 {
    RemainingSearchAllocationAfterTrim,
    SameContextFreeAfterAllocatorHeadroom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2CalibrationAxisV2 {
    DeviceUuid,
    PrimaryContext,
    SearchStream,
    ActivePool,
    CudaBuildIdentity,
    KernelSemanticsIdentity,
    Binary64MathIdentity,
    PlanIdentity,
    RunIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AdmissionErrorV2 {
    ImplementationPending,
    MissingArchiveArena,
    ZeroArchiveArenaBytes,
    AlignedLayoutFieldMismatch {
        field: ResidentSearchSlice2AlignedFieldV2,
        expected_aligned_bytes: u64,
        observed_aligned_bytes: u64,
    },
    ReceiptArithmeticOverflow {
        operation: ResidentSearchSlice2ReceiptArithmeticV2,
    },
    ReceiptTotalMismatch {
        axis: ResidentSearchSlice2ReceiptTotalAxisV2,
        expected_total_bytes: u64,
        observed_total_bytes: u64,
    },
    ReserveAuthorityBytesMismatch {
        authority: ResidentSearchSlice2ReserveAuthorityKindV2,
        expected_bytes: u64,
        observed_bytes: u64,
    },
    ReserveAuthorityBindingMismatch {
        authority: ResidentSearchSlice2ReserveAuthorityKindV2,
        axis: ResidentSearchSlice2AuthorityBindingAxisV2,
    },
    ReserveAuthorityRelationMismatch {
        relation: ResidentSearchSlice2ReserveRelationV2,
    },
    ReserveArithmeticOverflow {
        operation: ResidentSearchSlice2ReserveArithmeticV2,
    },
    InsufficientAllocationBudget {
        axis: ResidentSearchSlice2AllocationBudgetAxisV2,
        required_bytes: u64,
        available_bytes: u64,
    },
    ForeignCalibration {
        axis: ResidentSearchSlice2CalibrationAxisV2,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AllocationSymbolV2 {
    CudaHostAlloc,
    CudaMallocAsync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AllocationCategoryV2 {
    TerminalHostReceipt,
    GenerationArena,
    ScoringArchiveArena,
    ArchiveOnlyArena,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AllocationCallV2 {
    pub(crate) ordinal: u8,
    pub(crate) symbol: ResidentSearchSlice2AllocationSymbolV2,
    pub(crate) category: ResidentSearchSlice2AllocationCategoryV2,
    pub(crate) requested_bytes: u64,
    pub(crate) aligned_bytes: u64,
    pub(crate) alignment_bytes: u64,
    pub(crate) flags: u32,
    pub(crate) stream_identity: Option<u64>,
    pub(crate) pool_identity: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2HostAllocationArgsV2 {
    pub(crate) ordinal: u8,
    pub(crate) category: ResidentSearchSlice2AllocationCategoryV2,
    pub(crate) requested_bytes: u64,
    pub(crate) aligned_bytes: u64,
    pub(crate) alignment_bytes: u64,
    pub(crate) flags: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AsyncAllocationArgsV2 {
    pub(crate) ordinal: u8,
    pub(crate) category: ResidentSearchSlice2AllocationCategoryV2,
    pub(crate) requested_bytes: u64,
    pub(crate) aligned_bytes: u64,
    pub(crate) alignment_bytes: u64,
    pub(crate) flags: u32,
    pub(crate) stream_identity: u64,
    pub(crate) pool_identity: u64,
}

pub(crate) trait ResidentSearchSlice2AllocationFacadeV2 {
    fn begin_native_create(&mut self);
    fn cuda_host_alloc(&mut self, actual: ResidentSearchSlice2HostAllocationArgsV2);
    fn cuda_malloc_async(&mut self, actual: ResidentSearchSlice2AsyncAllocationArgsV2);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AlignedLayoutV2 {
    pub(crate) archive_gene_scalars: u64,
    pub(crate) archive_term_indices: u64,
    pub(crate) archive_term_weights: u64,
    pub(crate) archive_metric_rows: u64,
    pub(crate) archive_signatures: u64,
    pub(crate) archive_hashes: u64,
    pub(crate) current_population_signatures: u64,
    pub(crate) novelty_scores: u64,
    pub(crate) exact_top_k_keys: u64,
    pub(crate) admission_flags: u64,
    pub(crate) admission_offsets: u64,
    pub(crate) archive_control_and_seal: u64,
    pub(crate) replacement_subtotal_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2GenerationReceiptV2 {
    pub(crate) logical_gene_scalar_bytes: u64,
    pub(crate) logical_gene_index_bytes: u64,
    pub(crate) logical_gene_weight_bytes: u64,
    pub(crate) offspring_bytes: u64,
    pub(crate) metric_row_bytes: u64,
    pub(crate) rank_key_bytes: u64,
    pub(crate) selection_bytes: u64,
    pub(crate) dedup_hash_bytes: u64,
    pub(crate) cub_scratch_bytes: u64,
    pub(crate) retained_evaluation_workspace_bytes: u64,
    pub(crate) terminal_device_receipt_bytes: u64,
    pub(crate) total_device_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2ScoringArchiveReceiptV2 {
    pub(crate) fitness_score_bytes: u64,
    pub(crate) decision_key_bytes: u64,
    pub(crate) cub_scratch_bytes: u64,
    pub(crate) layout: ResidentSearchSlice2AlignedLayoutV2,
    pub(crate) total_device_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AuthorityBindingV2 {
    pub(crate) device_uuid: [u8; 16],
    pub(crate) primary_context_identity: u64,
    pub(crate) search_stream_identity: u64,
    pub(crate) active_pool_identity: u64,
    pub(crate) run_identity: u64,
    pub(crate) full_workspace_receipt_identity: u64,
    pub(crate) post_trim_receipt_identity: u64,
    pub(crate) authority_identity: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2ObservedReserveAuthorityV2 {
    pub(crate) bytes: u64,
    pub(crate) binding: ResidentSearchSlice2AuthorityBindingV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2ObservedReserveSetV2 {
    pub(crate) allocator_context_headroom: ResidentSearchSlice2ObservedReserveAuthorityV2,
    pub(crate) full_workspace_authority: ResidentSearchSlice2ObservedReserveAuthorityV2,
    pub(crate) retained_pre_search_workspace: ResidentSearchSlice2ObservedReserveAuthorityV2,
    pub(crate) remaining_search_allocation_after_trim:
        ResidentSearchSlice2ObservedReserveAuthorityV2,
    pub(crate) same_context_free: ResidentSearchSlice2ObservedReserveAuthorityV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2CalibrationBindingV2 {
    pub(crate) device_uuid: [u8; 16],
    pub(crate) primary_context_identity: u64,
    pub(crate) search_stream_identity: u64,
    pub(crate) active_pool_identity: u64,
    pub(crate) cuda_build_identity: u64,
    pub(crate) kernel_semantics_identity: u64,
    pub(crate) binary64_math_identity: u64,
    pub(crate) plan_identity: u64,
    pub(crate) run_identity: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2TrustedReserveAuthorityV2 {
    expected_bytes: u64,
    expected_binding: ResidentSearchSlice2AuthorityBindingV2,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2TrustedReserveSetV2 {
    allocator_context_headroom: ResidentSearchSlice2TrustedReserveAuthorityV2,
    full_workspace_authority: ResidentSearchSlice2TrustedReserveAuthorityV2,
    retained_pre_search_workspace: ResidentSearchSlice2TrustedReserveAuthorityV2,
    remaining_search_allocation_after_trim: ResidentSearchSlice2TrustedReserveAuthorityV2,
    same_context_free: ResidentSearchSlice2TrustedReserveAuthorityV2,
}

pub(crate) struct ResidentSearchSlice2TrustedReserveSealV2 {
    trusted_reserve: ResidentSearchSlice2TrustedReserveSetV2,
    expected_calibration: ResidentSearchSlice2CalibrationBindingV2,
    sealed_full_workspace_receipt_identity: u64,
    sealed_post_trim_receipt_identity: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AdmissionRequestV2 {
    pub(crate) population_count: u64,
    pub(crate) archive_capacity: u64,
    pub(crate) signature_word_count: u32,
    pub(crate) novelty_neighbor_count: u32,
    pub(crate) max_terms_per_gene: u32,
    pub(crate) terminal_host_receipt_bytes: u64,
    pub(crate) terminal_host_alignment_bytes: u64,
    pub(crate) device_alignment_bytes: u64,
    pub(crate) terminal_host_flags: u32,
    pub(crate) archive_arena_present: bool,
    pub(crate) archive_arena_bytes: u64,
    pub(crate) expected_slice2_layout: ResidentSearchSlice2AlignedLayoutV2,
    pub(crate) generation_receipt: ResidentSearchSlice2GenerationReceiptV2,
    pub(crate) scoring_archive_receipt: ResidentSearchSlice2ScoringArchiveReceiptV2,
    pub(crate) observed_reserve: ResidentSearchSlice2ObservedReserveSetV2,
    pub(crate) calibration: ResidentSearchSlice2CalibrationBindingV2,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AdmissionOwnerV2 {
    _move_only: (),
}

impl ResidentSearchSlice2AdmissionOwnerV2 {
    pub(crate) fn queue_generation_v2(
        self,
        _ordinal: u64,
        _allocator: &mut dyn ResidentSearchSlice2AllocationFacadeV2,
    ) -> Result<Self, ResidentSearchSlice2AdmissionErrorV2> {
        Err(ResidentSearchSlice2AdmissionErrorV2::ImplementationPending)
    }
}

pub(crate) fn admit_slice2_combined_fixture_v2(
    _request: ResidentSearchSlice2AdmissionRequestV2,
    _trusted_seal: ResidentSearchSlice2TrustedReserveSealV2,
    _allocator: &mut dyn ResidentSearchSlice2AllocationFacadeV2,
) -> Result<ResidentSearchSlice2AdmissionOwnerV2, ResidentSearchSlice2AdmissionErrorV2> {
    Err(ResidentSearchSlice2AdmissionErrorV2::ImplementationPending)
}

#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
#[path = "resident_search_v2_tests.rs"]
mod resident_search_v2_tests;
