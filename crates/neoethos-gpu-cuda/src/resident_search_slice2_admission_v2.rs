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

pub(crate) struct ResidentSearchSlice2ValidatedAdmissionV2 {
    terminal_host_receipt: ResidentSearchSlice2HostAllocationArgsV2,
    generation_arena: ResidentSearchSlice2AsyncAllocationArgsV2,
    scoring_archive_arena: ResidentSearchSlice2AsyncAllocationArgsV2,
}

impl ResidentSearchSlice2ValidatedAdmissionV2 {
    pub(crate) fn into_allocation_calls_v2(
        self,
    ) -> (
        ResidentSearchSlice2HostAllocationArgsV2,
        ResidentSearchSlice2AsyncAllocationArgsV2,
        ResidentSearchSlice2AsyncAllocationArgsV2,
    ) {
        (
            self.terminal_host_receipt,
            self.generation_arena,
            self.scoring_archive_arena,
        )
    }
}

impl ResidentSearchSlice2AdmissionOwnerV2 {
    pub(crate) fn queue_generation_v2(
        self,
        _ordinal: u64,
        _allocator: &mut dyn ResidentSearchSlice2AllocationFacadeV2,
    ) -> Result<Self, ResidentSearchSlice2AdmissionErrorV2> {
        Ok(self)
    }
}

fn checked_receipt_sum_v2(
    values: &[u64],
    operation: ResidentSearchSlice2ReceiptArithmeticV2,
) -> Result<u64, ResidentSearchSlice2AdmissionErrorV2> {
    values
        .iter()
        .copied()
        .try_fold(0_u64, u64::checked_add)
        .ok_or(ResidentSearchSlice2AdmissionErrorV2::ReceiptArithmeticOverflow { operation })
}

fn validate_reserve_authority_v2(
    authority: ResidentSearchSlice2ReserveAuthorityKindV2,
    observed: &ResidentSearchSlice2ObservedReserveAuthorityV2,
    trusted: &ResidentSearchSlice2TrustedReserveAuthorityV2,
    sealed_full_workspace_receipt_identity: u64,
    sealed_post_trim_receipt_identity: u64,
) -> Result<(), ResidentSearchSlice2AdmissionErrorV2> {
    if observed.bytes != trusted.expected_bytes {
        return Err(
            ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityBytesMismatch {
                authority,
                expected_bytes: trusted.expected_bytes,
                observed_bytes: observed.bytes,
            },
        );
    }

    let observed_binding = &observed.binding;
    let trusted_binding = &trusted.expected_binding;
    let mismatch = if observed_binding.device_uuid != trusted_binding.device_uuid {
        Some(ResidentSearchSlice2AuthorityBindingAxisV2::DeviceUuid)
    } else if observed_binding.primary_context_identity != trusted_binding.primary_context_identity
    {
        Some(ResidentSearchSlice2AuthorityBindingAxisV2::PrimaryContext)
    } else if observed_binding.search_stream_identity != trusted_binding.search_stream_identity {
        Some(ResidentSearchSlice2AuthorityBindingAxisV2::SearchStream)
    } else if observed_binding.active_pool_identity != trusted_binding.active_pool_identity {
        Some(ResidentSearchSlice2AuthorityBindingAxisV2::ActivePool)
    } else if observed_binding.run_identity != trusted_binding.run_identity {
        Some(ResidentSearchSlice2AuthorityBindingAxisV2::RunIdentity)
    } else if observed_binding.full_workspace_receipt_identity
        != trusted_binding.full_workspace_receipt_identity
        || trusted_binding.full_workspace_receipt_identity != sealed_full_workspace_receipt_identity
    {
        Some(ResidentSearchSlice2AuthorityBindingAxisV2::FullWorkspaceReceiptIdentity)
    } else if observed_binding.post_trim_receipt_identity
        != trusted_binding.post_trim_receipt_identity
        || trusted_binding.post_trim_receipt_identity != sealed_post_trim_receipt_identity
    {
        Some(ResidentSearchSlice2AuthorityBindingAxisV2::PostTrimReceiptIdentity)
    } else if observed_binding.authority_identity != trusted_binding.authority_identity {
        Some(ResidentSearchSlice2AuthorityBindingAxisV2::AuthorityIdentity)
    } else {
        None
    };

    if let Some(axis) = mismatch {
        return Err(
            ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityBindingMismatch {
                authority,
                axis,
            },
        );
    }
    Ok(())
}

pub(crate) fn validate_and_seal_slice2_combined_v2(
    request: ResidentSearchSlice2AdmissionRequestV2,
    trusted_seal: ResidentSearchSlice2TrustedReserveSealV2,
) -> Result<ResidentSearchSlice2ValidatedAdmissionV2, ResidentSearchSlice2AdmissionErrorV2> {
    let request = &request;
    let trusted_seal = &trusted_seal;
    if !request.archive_arena_present {
        return Err(ResidentSearchSlice2AdmissionErrorV2::MissingArchiveArena);
    }
    if request.archive_arena_bytes == 0 {
        return Err(ResidentSearchSlice2AdmissionErrorV2::ZeroArchiveArenaBytes);
    }

    let expected_layout = &request.expected_slice2_layout;
    let observed_layout = &request.scoring_archive_receipt.layout;
    let aligned_fields = [
        (
            ResidentSearchSlice2AlignedFieldV2::ArchiveGeneScalars,
            expected_layout.archive_gene_scalars,
            observed_layout.archive_gene_scalars,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::ArchiveTermIndices,
            expected_layout.archive_term_indices,
            observed_layout.archive_term_indices,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::ArchiveTermWeights,
            expected_layout.archive_term_weights,
            observed_layout.archive_term_weights,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::ArchiveMetricRows,
            expected_layout.archive_metric_rows,
            observed_layout.archive_metric_rows,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::ArchiveSignatures,
            expected_layout.archive_signatures,
            observed_layout.archive_signatures,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::ArchiveHashes,
            expected_layout.archive_hashes,
            observed_layout.archive_hashes,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::CurrentPopulationSignatures,
            expected_layout.current_population_signatures,
            observed_layout.current_population_signatures,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::NoveltyScores,
            expected_layout.novelty_scores,
            observed_layout.novelty_scores,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::ExactTopKKeys,
            expected_layout.exact_top_k_keys,
            observed_layout.exact_top_k_keys,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::AdmissionFlags,
            expected_layout.admission_flags,
            observed_layout.admission_flags,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::AdmissionOffsets,
            expected_layout.admission_offsets,
            observed_layout.admission_offsets,
        ),
        (
            ResidentSearchSlice2AlignedFieldV2::ArchiveControlAndSeal,
            expected_layout.archive_control_and_seal,
            observed_layout.archive_control_and_seal,
        ),
    ];
    for (field, expected_aligned_bytes, observed_aligned_bytes) in aligned_fields {
        if observed_aligned_bytes != expected_aligned_bytes {
            return Err(
                ResidentSearchSlice2AdmissionErrorV2::AlignedLayoutFieldMismatch {
                    field,
                    expected_aligned_bytes,
                    observed_aligned_bytes,
                },
            );
        }
    }

    let replacement_subtotal = checked_receipt_sum_v2(
        &[
            observed_layout.archive_gene_scalars,
            observed_layout.archive_term_indices,
            observed_layout.archive_term_weights,
            observed_layout.archive_metric_rows,
            observed_layout.archive_signatures,
            observed_layout.archive_hashes,
            observed_layout.current_population_signatures,
            observed_layout.novelty_scores,
            observed_layout.exact_top_k_keys,
            observed_layout.admission_flags,
            observed_layout.admission_offsets,
            observed_layout.archive_control_and_seal,
        ],
        ResidentSearchSlice2ReceiptArithmeticV2::ReplacementSubtotalAdd,
    )?;
    if observed_layout.replacement_subtotal_bytes != replacement_subtotal {
        return Err(ResidentSearchSlice2AdmissionErrorV2::ReceiptTotalMismatch {
            axis: ResidentSearchSlice2ReceiptTotalAxisV2::ReplacementSubtotal,
            expected_total_bytes: replacement_subtotal,
            observed_total_bytes: observed_layout.replacement_subtotal_bytes,
        });
    }

    let generation = &request.generation_receipt;
    let generation_total = checked_receipt_sum_v2(
        &[
            generation.logical_gene_scalar_bytes,
            generation.logical_gene_index_bytes,
            generation.logical_gene_weight_bytes,
            generation.offspring_bytes,
            generation.metric_row_bytes,
            generation.rank_key_bytes,
            generation.selection_bytes,
            generation.dedup_hash_bytes,
            generation.cub_scratch_bytes,
            generation.retained_evaluation_workspace_bytes,
            generation.terminal_device_receipt_bytes,
        ],
        ResidentSearchSlice2ReceiptArithmeticV2::GenerationReceiptTotalAdd,
    )?;
    if generation.total_device_bytes != generation_total {
        return Err(ResidentSearchSlice2AdmissionErrorV2::ReceiptTotalMismatch {
            axis: ResidentSearchSlice2ReceiptTotalAxisV2::GenerationReceiptTotal,
            expected_total_bytes: generation_total,
            observed_total_bytes: generation.total_device_bytes,
        });
    }

    let scoring = &request.scoring_archive_receipt;
    let scoring_total = checked_receipt_sum_v2(
        &[
            scoring.fitness_score_bytes,
            scoring.decision_key_bytes,
            scoring.cub_scratch_bytes,
            observed_layout.archive_gene_scalars,
            observed_layout.archive_term_indices,
            observed_layout.archive_term_weights,
            observed_layout.archive_metric_rows,
            observed_layout.archive_signatures,
            observed_layout.archive_hashes,
            observed_layout.current_population_signatures,
            observed_layout.novelty_scores,
            observed_layout.exact_top_k_keys,
            observed_layout.admission_flags,
            observed_layout.admission_offsets,
            observed_layout.archive_control_and_seal,
        ],
        ResidentSearchSlice2ReceiptArithmeticV2::ScoringArchiveReceiptTotalAdd,
    )?;
    if scoring.total_device_bytes != scoring_total {
        return Err(ResidentSearchSlice2AdmissionErrorV2::ReceiptTotalMismatch {
            axis: ResidentSearchSlice2ReceiptTotalAxisV2::ScoringArchiveReceiptTotal,
            expected_total_bytes: scoring_total,
            observed_total_bytes: scoring.total_device_bytes,
        });
    }

    let observed_reserve = &request.observed_reserve;
    let trusted_reserve = &trusted_seal.trusted_reserve;
    for (authority, observed, trusted) in [
        (
            ResidentSearchSlice2ReserveAuthorityKindV2::AllocatorContextHeadroom,
            &observed_reserve.allocator_context_headroom,
            &trusted_reserve.allocator_context_headroom,
        ),
        (
            ResidentSearchSlice2ReserveAuthorityKindV2::FullWorkspaceAuthority,
            &observed_reserve.full_workspace_authority,
            &trusted_reserve.full_workspace_authority,
        ),
        (
            ResidentSearchSlice2ReserveAuthorityKindV2::RetainedPreSearchWorkspace,
            &observed_reserve.retained_pre_search_workspace,
            &trusted_reserve.retained_pre_search_workspace,
        ),
        (
            ResidentSearchSlice2ReserveAuthorityKindV2::RemainingSearchAllocationAfterTrim,
            &observed_reserve.remaining_search_allocation_after_trim,
            &trusted_reserve.remaining_search_allocation_after_trim,
        ),
        (
            ResidentSearchSlice2ReserveAuthorityKindV2::SameContextFree,
            &observed_reserve.same_context_free,
            &trusted_reserve.same_context_free,
        ),
    ] {
        validate_reserve_authority_v2(
            authority,
            observed,
            trusted,
            trusted_seal.sealed_full_workspace_receipt_identity,
            trusted_seal.sealed_post_trim_receipt_identity,
        )?;
    }

    let authority_identities = [
        observed_reserve
            .allocator_context_headroom
            .binding
            .authority_identity,
        observed_reserve
            .full_workspace_authority
            .binding
            .authority_identity,
        observed_reserve
            .retained_pre_search_workspace
            .binding
            .authority_identity,
        observed_reserve
            .remaining_search_allocation_after_trim
            .binding
            .authority_identity,
    ];
    for left in 0..authority_identities.len() {
        for right in (left + 1)..authority_identities.len() {
            if authority_identities[left] == authority_identities[right] {
                return Err(
                    ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityRelationMismatch {
                        relation: ResidentSearchSlice2ReserveRelationV2::FourReserveAuthorityIdentitiesDistinct,
                    },
                );
            }
        }
    }

    let workspace_partition = observed_reserve
        .retained_pre_search_workspace
        .bytes
        .checked_add(
            observed_reserve
                .remaining_search_allocation_after_trim
                .bytes,
        )
        .ok_or(
            ResidentSearchSlice2AdmissionErrorV2::ReserveArithmeticOverflow {
                operation: ResidentSearchSlice2ReserveArithmeticV2::WorkspacePartitionAdd,
            },
        )?;
    if workspace_partition != observed_reserve.full_workspace_authority.bytes {
        return Err(
            ResidentSearchSlice2AdmissionErrorV2::ReserveAuthorityRelationMismatch {
                relation:
                    ResidentSearchSlice2ReserveRelationV2::RetainedPlusRemainingEqualsFullWorkspace,
            },
        );
    }

    let requested_device_bytes = generation
        .total_device_bytes
        .checked_add(scoring.total_device_bytes)
        .ok_or(
            ResidentSearchSlice2AdmissionErrorV2::ReserveArithmeticOverflow {
                operation: ResidentSearchSlice2ReserveArithmeticV2::RequestedDeviceSumAdd,
            },
        )?;
    let same_context_free_after_headroom = observed_reserve
        .same_context_free
        .bytes
        .checked_sub(observed_reserve.allocator_context_headroom.bytes)
        .ok_or(
            ResidentSearchSlice2AdmissionErrorV2::ReserveArithmeticOverflow {
                operation: ResidentSearchSlice2ReserveArithmeticV2::SameContextFreeMinusHeadroom,
            },
        )?;
    let remaining_bytes = observed_reserve
        .remaining_search_allocation_after_trim
        .bytes;
    if remaining_bytes < requested_device_bytes {
        return Err(
            ResidentSearchSlice2AdmissionErrorV2::InsufficientAllocationBudget {
                axis:
                    ResidentSearchSlice2AllocationBudgetAxisV2::RemainingSearchAllocationAfterTrim,
                required_bytes: requested_device_bytes,
                available_bytes: remaining_bytes,
            },
        );
    }
    if same_context_free_after_headroom < requested_device_bytes {
        return Err(ResidentSearchSlice2AdmissionErrorV2::InsufficientAllocationBudget {
            axis: ResidentSearchSlice2AllocationBudgetAxisV2::SameContextFreeAfterAllocatorHeadroom,
            required_bytes: requested_device_bytes,
            available_bytes: same_context_free_after_headroom,
        });
    }

    let calibration = &request.calibration;
    let expected_calibration = &trusted_seal.expected_calibration;
    let foreign_axis = if calibration.device_uuid != expected_calibration.device_uuid {
        Some(ResidentSearchSlice2CalibrationAxisV2::DeviceUuid)
    } else if calibration.primary_context_identity != expected_calibration.primary_context_identity
    {
        Some(ResidentSearchSlice2CalibrationAxisV2::PrimaryContext)
    } else if calibration.search_stream_identity != expected_calibration.search_stream_identity {
        Some(ResidentSearchSlice2CalibrationAxisV2::SearchStream)
    } else if calibration.active_pool_identity != expected_calibration.active_pool_identity {
        Some(ResidentSearchSlice2CalibrationAxisV2::ActivePool)
    } else if calibration.cuda_build_identity != expected_calibration.cuda_build_identity {
        Some(ResidentSearchSlice2CalibrationAxisV2::CudaBuildIdentity)
    } else if calibration.kernel_semantics_identity
        != expected_calibration.kernel_semantics_identity
    {
        Some(ResidentSearchSlice2CalibrationAxisV2::KernelSemanticsIdentity)
    } else if calibration.binary64_math_identity != expected_calibration.binary64_math_identity {
        Some(ResidentSearchSlice2CalibrationAxisV2::Binary64MathIdentity)
    } else if calibration.plan_identity != expected_calibration.plan_identity {
        Some(ResidentSearchSlice2CalibrationAxisV2::PlanIdentity)
    } else if calibration.run_identity != expected_calibration.run_identity {
        Some(ResidentSearchSlice2CalibrationAxisV2::RunIdentity)
    } else {
        None
    };
    if let Some(axis) = foreign_axis {
        return Err(ResidentSearchSlice2AdmissionErrorV2::ForeignCalibration { axis });
    }

    Ok(ResidentSearchSlice2ValidatedAdmissionV2 {
        terminal_host_receipt: ResidentSearchSlice2HostAllocationArgsV2 {
            ordinal: 0,
            category: ResidentSearchSlice2AllocationCategoryV2::TerminalHostReceipt,
            requested_bytes: request.terminal_host_receipt_bytes,
            aligned_bytes: request.terminal_host_receipt_bytes,
            alignment_bytes: request.terminal_host_alignment_bytes,
            flags: request.terminal_host_flags,
        },
        generation_arena: ResidentSearchSlice2AsyncAllocationArgsV2 {
            ordinal: 1,
            category: ResidentSearchSlice2AllocationCategoryV2::GenerationArena,
            requested_bytes: generation.total_device_bytes,
            aligned_bytes: generation.total_device_bytes,
            alignment_bytes: request.device_alignment_bytes,
            flags: 0,
            stream_identity: calibration.search_stream_identity,
            pool_identity: calibration.active_pool_identity,
        },
        scoring_archive_arena: ResidentSearchSlice2AsyncAllocationArgsV2 {
            ordinal: 2,
            category: ResidentSearchSlice2AllocationCategoryV2::ScoringArchiveArena,
            requested_bytes: scoring.total_device_bytes,
            aligned_bytes: scoring.total_device_bytes,
            alignment_bytes: request.device_alignment_bytes,
            flags: 0,
            stream_identity: calibration.search_stream_identity,
            pool_identity: calibration.active_pool_identity,
        },
    })
}

pub(crate) fn admit_slice2_combined_fixture_v2(
    request: ResidentSearchSlice2AdmissionRequestV2,
    _trusted_seal: ResidentSearchSlice2TrustedReserveSealV2,
    allocator: &mut dyn ResidentSearchSlice2AllocationFacadeV2,
) -> Result<ResidentSearchSlice2AdmissionOwnerV2, ResidentSearchSlice2AdmissionErrorV2> {
    let validated = validate_and_seal_slice2_combined_v2(request, _trusted_seal)?;
    let (host_receipt, generation_arena, scoring_archive_arena) =
        validated.into_allocation_calls_v2();
    allocator.begin_native_create();
    allocator.cuda_host_alloc(host_receipt);
    allocator.cuda_malloc_async(generation_arena);
    allocator.cuda_malloc_async(scoring_archive_arena);
    Ok(ResidentSearchSlice2AdmissionOwnerV2 { _move_only: () })
}

#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
#[path = "resident_search_v2_tests.rs"]
mod resident_search_v2_tests;
