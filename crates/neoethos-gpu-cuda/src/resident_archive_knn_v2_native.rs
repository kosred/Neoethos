use super::{
    ResidentSearchSlice2AllocationCategoryV2, ResidentSearchSlice2AsyncAllocationArgsV2,
    ResidentSearchSlice2CalibrationBindingV2, ResidentSearchSlice2ScoringArchiveReceiptV2,
};
#[cfg(feature = "cuda")]
use crate::population::RawResidentScoringPopulationSourceV2;
#[cfg(feature = "cuda")]
use crate::resident_generation_v1::{NativeResidentGenerationRunV1, RawReadyEventV1};
#[cfg(feature = "cuda")]
use crate::resident_search_v2::RawResidentGenerationGeneViewV2;
#[cfg(feature = "cuda")]
use std::ffi::c_void;

const SCORING_ARCHIVE_ALIGNMENT_BYTES_V2: u64 = 256;
const FITNESS_SCORE_BYTES_V2: u64 = 1_792;
const DECISION_KEY_BYTES_V2: u64 = 1_792;
const REPLACEMENT_SUBTOTAL_BYTES_V2: u64 = 23_707_648;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RawResidentArchiveKnnArenaRegionV2 {
    pub(super) offset_bytes: u64,
    pub(super) size_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RawResidentArchiveKnnBindV2 {
    pub(super) abi_version: u32,
    pub(super) reserved: u32,
    pub(super) fitness_scores: RawResidentArchiveKnnArenaRegionV2,
    pub(super) decision_keys: RawResidentArchiveKnnArenaRegionV2,
    pub(super) cub_scratch: RawResidentArchiveKnnArenaRegionV2,
    pub(super) archive_gene_scalars: RawResidentArchiveKnnArenaRegionV2,
    pub(super) archive_term_indices: RawResidentArchiveKnnArenaRegionV2,
    pub(super) archive_term_weights: RawResidentArchiveKnnArenaRegionV2,
    pub(super) archive_metric_rows: RawResidentArchiveKnnArenaRegionV2,
    pub(super) archive_signatures: RawResidentArchiveKnnArenaRegionV2,
    pub(super) archive_hashes: RawResidentArchiveKnnArenaRegionV2,
    pub(super) current_population_signatures: RawResidentArchiveKnnArenaRegionV2,
    pub(super) novelty_scores: RawResidentArchiveKnnArenaRegionV2,
    pub(super) exact_top_k_keys: RawResidentArchiveKnnArenaRegionV2,
    pub(super) admission_flags: RawResidentArchiveKnnArenaRegionV2,
    pub(super) admission_offsets: RawResidentArchiveKnnArenaRegionV2,
    pub(super) archive_control_and_seal: RawResidentArchiveKnnArenaRegionV2,
    pub(super) total_device_bytes: u64,
    pub(super) population_count: u64,
    pub(super) archive_capacity: u64,
    pub(super) signature_word_count: u32,
    pub(super) novelty_neighbor_count: u32,
    pub(super) max_terms_per_gene: u32,
    pub(super) reserved_extents: u32,
    pub(super) device_uuid: [u8; 16],
    pub(super) primary_context_identity: u64,
    pub(super) search_stream_identity: u64,
    pub(super) active_pool_identity: u64,
    pub(super) cuda_build_identity: u64,
    pub(super) kernel_semantics_identity: u64,
    pub(super) binary64_math_identity: u64,
    pub(super) plan_identity: u64,
    pub(super) run_identity: u64,
    pub(super) full_workspace_receipt_identity: u64,
    pub(super) post_trim_receipt_identity: u64,
}

#[cfg(feature = "cuda")]
pub(crate) enum NativeResidentScoringNoveltyRunV1 {}
#[cfg(feature = "cuda")]
pub(crate) enum NativeResidentArchiveKnnOwnerV2 {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RawResidentArchiveKnnPendingV2 {
    pub(super) abi_version: u32,
    pub(super) flags: u32,
    pub(super) source_packed_commit_word: u64,
    pub(super) terminal_device_receipt_identity: u64,
    pub(super) run_identity: u64,
    pub(super) boxed_receipt_identity: u64,
    pub(super) staged_dependency_identity: u64,
    pub(super) same_stream_enqueue_count: u64,
    pub(super) completion_event_identity: u64,
    pub(super) terminal_host_receipt_identity: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RawResidentArchiveKnnTerminalV2 {
    pub(super) abi_version: u32,
    pub(super) terminal_status: u32,
    pub(super) device_fault_word: u32,
    pub(super) validation_fault_word: u32,
    pub(super) receipt_identity: u64,
    pub(super) run_identity: u64,
    pub(super) packed_commit_word: u64,
    pub(super) collision_count: u64,
    pub(super) compact_async_d2h_count: u64,
    pub(super) compact_async_d2h_bytes: u64,
    pub(super) completion_event_query_count: u64,
    pub(super) completion_stream_synchronize_count: u64,
    pub(super) same_stream_enqueue_count: u64,
    pub(super) completion_event_identity: u64,
    pub(super) validator_digest: u64,
}

#[cfg(feature = "cuda")]
impl RawResidentArchiveKnnTerminalV2 {
    pub(crate) fn validates_committed_v2(
        &self,
        pending: &RawResidentArchiveKnnPendingV2,
        binding: &RawResidentArchiveKnnBindV2,
        ready: &RawReadyEventV1,
    ) -> bool {
        let generation = (self.packed_commit_word >> 1) & 0xffff;
        let archive_count = (self.packed_commit_word >> 17) & 0xffff;
        let mut digest = 1_469_598_103_934_665_603_u64;
        for lane in [
            self.packed_commit_word,
            self.collision_count,
            binding.run_identity,
            u64::from(self.device_fault_word),
        ] {
            for byte in lane.to_le_bytes() {
                digest ^= u64::from(byte);
                digest = digest.wrapping_mul(1_099_511_628_211);
            }
        }
        self.abi_version == 2
            && self.terminal_status == 1
            && self.device_fault_word == 0
            && self.validation_fault_word == 0
            && self.receipt_identity == pending.terminal_host_receipt_identity
            && self.run_identity == binding.run_identity
            && self.run_identity == pending.run_identity
            && archive_count <= binding.archive_capacity
            && self.compact_async_d2h_count == 1
            && self.compact_async_d2h_bytes == std::mem::size_of::<Self>() as u64
            && self.completion_event_query_count != 0
            && self.completion_stream_synchronize_count == 0
            && self.same_stream_enqueue_count == pending.same_stream_enqueue_count
            && self.completion_event_identity == pending.completion_event_identity
            && self.validator_digest == digest
            && ready.abi_version == 1
            && ready.reserved == 0
            && ready.event_id == pending.completion_event_identity
            && ready.generation_index == generation
            && ready.same_stream_enqueue_count == pending.same_stream_enqueue_count
            && ready.intermediate_host_wait_count == 0
            && ready.intermediate_readback_count == 0
    }
}

const _: [(); 16] = [(); std::mem::size_of::<RawResidentArchiveKnnArenaRegionV2>()];
const _: [(); 384] = [(); std::mem::size_of::<RawResidentArchiveKnnBindV2>()];
const _: [(); 72] = [(); std::mem::size_of::<RawResidentArchiveKnnPendingV2>()];
const _: [(); 8] = [(); std::mem::align_of::<RawResidentArchiveKnnPendingV2>()];
const _: [(); 104] = [(); std::mem::size_of::<RawResidentArchiveKnnTerminalV2>()];
const _: [(); 8] = [(); std::mem::align_of::<RawResidentArchiveKnnTerminalV2>()];

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub(crate) fn bind_preallocated_resident_archive_knn_v2(
        scoring: *mut NativeResidentScoringNoveltyRunV1,
        generation: *mut NativeResidentGenerationRunV1,
        genes: *const RawResidentGenerationGeneViewV2,
        binding: *const RawResidentArchiveKnnBindV2,
        owner: *mut *mut NativeResidentArchiveKnnOwnerV2,
    ) -> i32;

    pub(crate) fn enqueue_resident_archive_score_and_rank_v2(
        owner: *mut NativeResidentArchiveKnnOwnerV2,
        population: *const RawResidentScoringPopulationSourceV2,
        dependency: *const RawReadyEventV1,
    ) -> i32;

    pub(crate) fn enqueue_resident_archive_stage_from_rank_v2(
        owner: *mut NativeResidentArchiveKnnOwnerV2,
    ) -> i32;

    pub(crate) fn enqueue_resident_archive_evolve_and_publish_v2(
        owner: *mut NativeResidentArchiveKnnOwnerV2,
    ) -> i32;

    pub(crate) fn enqueue_resident_archive_terminal_seal_v2(
        owner: *mut NativeResidentArchiveKnnOwnerV2,
        pending: *mut RawResidentArchiveKnnPendingV2,
    ) -> i32;

    pub(crate) fn try_complete_resident_archive_terminal_v2(
        owner: *mut NativeResidentArchiveKnnOwnerV2,
        pending: *const RawResidentArchiveKnnPendingV2,
        committed_ready: *mut RawReadyEventV1,
        terminal_copy: *mut RawResidentArchiveKnnTerminalV2,
    ) -> i32;

    pub(crate) fn neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2(
        session: *mut c_void,
        owner: *mut NativeResidentArchiveKnnOwnerV2,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScoringArchiveArenaRegionV2 {
    FitnessScores,
    DecisionKeys,
    CubScratch,
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
struct RegionV2 {
    offset_bytes: u64,
    size_bytes: u64,
}

impl RegionV2 {
    const fn new(offset_bytes: u64, size_bytes: u64) -> Self {
        Self {
            offset_bytes,
            size_bytes,
        }
    }

    #[cfg(test)]
    const fn end_v2(self) -> Option<u64> {
        self.offset_bytes.checked_add(self.size_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScoringArchiveArenaLayoutErrorV2 {
    AllocationOrdinalMismatch {
        observed: u8,
    },
    AllocationCategoryMismatch,
    AllocationAlignmentMismatch {
        observed: u64,
    },
    AllocationFlagsMismatch {
        observed: u32,
    },
    CubScratchAlignmentMismatch {
        observed: u64,
    },
    RegionSizeMismatch {
        region: ScoringArchiveArenaRegionV2,
        expected: u64,
        observed: u64,
    },
    ReplacementSubtotalMismatch {
        expected: u64,
        observed: u64,
    },
    ArithmeticOverflow {
        region: ScoringArchiveArenaRegionV2,
    },
    ReceiptTotalMismatch {
        expected: u64,
        observed: u64,
    },
    AllocationTotalMismatch {
        expected: u64,
        observed_requested: u64,
        observed_aligned: u64,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ResidentScoringArchiveArenaLayoutV2 {
    fitness_scores: RegionV2,
    decision_keys: RegionV2,
    cub_scratch: RegionV2,
    archive_gene_scalars: RegionV2,
    archive_term_indices: RegionV2,
    archive_term_weights: RegionV2,
    archive_metric_rows: RegionV2,
    archive_signatures: RegionV2,
    archive_hashes: RegionV2,
    current_population_signatures: RegionV2,
    novelty_scores: RegionV2,
    exact_top_k_keys: RegionV2,
    admission_flags: RegionV2,
    admission_offsets: RegionV2,
    archive_control_and_seal: RegionV2,
    replacement_subtotal_bytes: u64,
    total_device_bytes: u64,
    stream_identity: u64,
    pool_identity: u64,
}

impl ResidentScoringArchiveArenaLayoutV2 {
    pub(super) fn into_native_bind_v2(
        self,
        calibration: ResidentSearchSlice2CalibrationBindingV2,
        population_count: u64,
        archive_capacity: u64,
        signature_word_count: u32,
        novelty_neighbor_count: u32,
        max_terms_per_gene: u32,
        full_workspace_receipt_identity: u64,
        post_trim_receipt_identity: u64,
    ) -> RawResidentArchiveKnnBindV2 {
        fn raw(region: RegionV2) -> RawResidentArchiveKnnArenaRegionV2 {
            RawResidentArchiveKnnArenaRegionV2 {
                offset_bytes: region.offset_bytes,
                size_bytes: region.size_bytes,
            }
        }

        RawResidentArchiveKnnBindV2 {
            abi_version: 2,
            reserved: 0,
            fitness_scores: raw(self.fitness_scores),
            decision_keys: raw(self.decision_keys),
            cub_scratch: raw(self.cub_scratch),
            archive_gene_scalars: raw(self.archive_gene_scalars),
            archive_term_indices: raw(self.archive_term_indices),
            archive_term_weights: raw(self.archive_term_weights),
            archive_metric_rows: raw(self.archive_metric_rows),
            archive_signatures: raw(self.archive_signatures),
            archive_hashes: raw(self.archive_hashes),
            current_population_signatures: raw(self.current_population_signatures),
            novelty_scores: raw(self.novelty_scores),
            exact_top_k_keys: raw(self.exact_top_k_keys),
            admission_flags: raw(self.admission_flags),
            admission_offsets: raw(self.admission_offsets),
            archive_control_and_seal: raw(self.archive_control_and_seal),
            total_device_bytes: self.total_device_bytes,
            population_count,
            archive_capacity,
            signature_word_count,
            novelty_neighbor_count,
            max_terms_per_gene,
            reserved_extents: 0,
            device_uuid: calibration.device_uuid,
            primary_context_identity: calibration.primary_context_identity,
            search_stream_identity: calibration.search_stream_identity,
            active_pool_identity: calibration.active_pool_identity,
            cuda_build_identity: calibration.cuda_build_identity,
            kernel_semantics_identity: calibration.kernel_semantics_identity,
            binary64_math_identity: calibration.binary64_math_identity,
            plan_identity: calibration.plan_identity,
            run_identity: calibration.run_identity,
            full_workspace_receipt_identity,
            post_trim_receipt_identity,
        }
    }

    #[cfg(test)]
    fn regions_v2(&self) -> [RegionV2; 15] {
        [
            self.fitness_scores,
            self.decision_keys,
            self.cub_scratch,
            self.archive_gene_scalars,
            self.archive_term_indices,
            self.archive_term_weights,
            self.archive_metric_rows,
            self.archive_signatures,
            self.archive_hashes,
            self.current_population_signatures,
            self.novelty_scores,
            self.exact_top_k_keys,
            self.admission_flags,
            self.admission_offsets,
            self.archive_control_and_seal,
        ]
    }

    #[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
    pub(super) fn test_archive_gene_scalars_v2(&self) -> (u64, u64) {
        (
            self.archive_gene_scalars.offset_bytes,
            self.archive_gene_scalars.size_bytes,
        )
    }

    #[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
    pub(super) fn test_total_device_bytes_v2(&self) -> u64 {
        self.total_device_bytes
    }
}

fn require_region_size_v2(
    region: ScoringArchiveArenaRegionV2,
    observed: u64,
    expected: u64,
) -> Result<(), ScoringArchiveArenaLayoutErrorV2> {
    if observed == expected {
        Ok(())
    } else {
        Err(ScoringArchiveArenaLayoutErrorV2::RegionSizeMismatch {
            region,
            expected,
            observed,
        })
    }
}

fn append_region_v2(
    cursor: &mut u64,
    region: ScoringArchiveArenaRegionV2,
    size_bytes: u64,
) -> Result<RegionV2, ScoringArchiveArenaLayoutErrorV2> {
    let offset_bytes = *cursor;
    *cursor = cursor
        .checked_add(size_bytes)
        .ok_or(ScoringArchiveArenaLayoutErrorV2::ArithmeticOverflow { region })?;
    Ok(RegionV2::new(offset_bytes, size_bytes))
}

pub(super) fn validate_scoring_archive_arena_layout_v2(
    allocation: ResidentSearchSlice2AsyncAllocationArgsV2,
    receipt: ResidentSearchSlice2ScoringArchiveReceiptV2,
) -> Result<ResidentScoringArchiveArenaLayoutV2, ScoringArchiveArenaLayoutErrorV2> {
    if allocation.ordinal != 2 {
        return Err(
            ScoringArchiveArenaLayoutErrorV2::AllocationOrdinalMismatch {
                observed: allocation.ordinal,
            },
        );
    }
    if allocation.category != ResidentSearchSlice2AllocationCategoryV2::ScoringArchiveArena {
        return Err(ScoringArchiveArenaLayoutErrorV2::AllocationCategoryMismatch);
    }
    if allocation.alignment_bytes != SCORING_ARCHIVE_ALIGNMENT_BYTES_V2 {
        return Err(
            ScoringArchiveArenaLayoutErrorV2::AllocationAlignmentMismatch {
                observed: allocation.alignment_bytes,
            },
        );
    }
    if allocation.flags != 0 {
        return Err(ScoringArchiveArenaLayoutErrorV2::AllocationFlagsMismatch {
            observed: allocation.flags,
        });
    }
    if receipt.cub_scratch_bytes % SCORING_ARCHIVE_ALIGNMENT_BYTES_V2 != 0 {
        return Err(
            ScoringArchiveArenaLayoutErrorV2::CubScratchAlignmentMismatch {
                observed: receipt.cub_scratch_bytes,
            },
        );
    }

    require_region_size_v2(
        ScoringArchiveArenaRegionV2::FitnessScores,
        receipt.fitness_score_bytes,
        FITNESS_SCORE_BYTES_V2,
    )?;
    require_region_size_v2(
        ScoringArchiveArenaRegionV2::DecisionKeys,
        receipt.decision_key_bytes,
        DECISION_KEY_BYTES_V2,
    )?;

    let expected_regions = [
        (
            ScoringArchiveArenaRegionV2::ArchiveGeneScalars,
            receipt.layout.archive_gene_scalars,
            3_600_128,
        ),
        (
            ScoringArchiveArenaRegionV2::ArchiveTermIndices,
            receipt.layout.archive_term_indices,
            6_400_000,
        ),
        (
            ScoringArchiveArenaRegionV2::ArchiveTermWeights,
            receipt.layout.archive_term_weights,
            6_400_000,
        ),
        (
            ScoringArchiveArenaRegionV2::ArchiveMetricRows,
            receipt.layout.archive_metric_rows,
            5_200_128,
        ),
        (
            ScoringArchiveArenaRegionV2::ArchiveSignatures,
            receipt.layout.archive_signatures,
            1_600_000,
        ),
        (
            ScoringArchiveArenaRegionV2::ArchiveHashes,
            receipt.layout.archive_hashes,
            400_128,
        ),
        (
            ScoringArchiveArenaRegionV2::CurrentPopulationSignatures,
            receipt.layout.current_population_signatures,
            6_400,
        ),
        (
            ScoringArchiveArenaRegionV2::NoveltyScores,
            receipt.layout.novelty_scores,
            1_792,
        ),
        (
            ScoringArchiveArenaRegionV2::ExactTopKKeys,
            receipt.layout.exact_top_k_keys,
            96_000,
        ),
        (
            ScoringArchiveArenaRegionV2::AdmissionFlags,
            receipt.layout.admission_flags,
            1_024,
        ),
        (
            ScoringArchiveArenaRegionV2::AdmissionOffsets,
            receipt.layout.admission_offsets,
            1_792,
        ),
        (
            ScoringArchiveArenaRegionV2::ArchiveControlAndSeal,
            receipt.layout.archive_control_and_seal,
            256,
        ),
    ];
    for (region, observed, expected) in expected_regions {
        require_region_size_v2(region, observed, expected)?;
    }

    let replacement_subtotal_bytes =
        expected_regions
            .iter()
            .try_fold(0_u64, |subtotal, (region, observed, _)| {
                subtotal
                    .checked_add(*observed)
                    .ok_or(ScoringArchiveArenaLayoutErrorV2::ArithmeticOverflow { region: *region })
            })?;
    if replacement_subtotal_bytes != REPLACEMENT_SUBTOTAL_BYTES_V2 {
        return Err(
            ScoringArchiveArenaLayoutErrorV2::ReplacementSubtotalMismatch {
                expected: REPLACEMENT_SUBTOTAL_BYTES_V2,
                observed: replacement_subtotal_bytes,
            },
        );
    }
    if receipt.layout.replacement_subtotal_bytes != replacement_subtotal_bytes {
        return Err(
            ScoringArchiveArenaLayoutErrorV2::ReplacementSubtotalMismatch {
                expected: replacement_subtotal_bytes,
                observed: receipt.layout.replacement_subtotal_bytes,
            },
        );
    }

    let mut cursor = 0_u64;
    let fitness_scores = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::FitnessScores,
        receipt.fitness_score_bytes,
    )?;
    let decision_keys = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::DecisionKeys,
        receipt.decision_key_bytes,
    )?;
    let cub_scratch = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::CubScratch,
        receipt.cub_scratch_bytes,
    )?;
    let archive_gene_scalars = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::ArchiveGeneScalars,
        receipt.layout.archive_gene_scalars,
    )?;
    let archive_term_indices = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::ArchiveTermIndices,
        receipt.layout.archive_term_indices,
    )?;
    let archive_term_weights = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::ArchiveTermWeights,
        receipt.layout.archive_term_weights,
    )?;
    let archive_metric_rows = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::ArchiveMetricRows,
        receipt.layout.archive_metric_rows,
    )?;
    let archive_signatures = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::ArchiveSignatures,
        receipt.layout.archive_signatures,
    )?;
    let archive_hashes = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::ArchiveHashes,
        receipt.layout.archive_hashes,
    )?;
    let current_population_signatures = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::CurrentPopulationSignatures,
        receipt.layout.current_population_signatures,
    )?;
    let novelty_scores = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::NoveltyScores,
        receipt.layout.novelty_scores,
    )?;
    let exact_top_k_keys = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::ExactTopKKeys,
        receipt.layout.exact_top_k_keys,
    )?;
    let admission_flags = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::AdmissionFlags,
        receipt.layout.admission_flags,
    )?;
    let admission_offsets = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::AdmissionOffsets,
        receipt.layout.admission_offsets,
    )?;
    let archive_control_and_seal = append_region_v2(
        &mut cursor,
        ScoringArchiveArenaRegionV2::ArchiveControlAndSeal,
        receipt.layout.archive_control_and_seal,
    )?;

    if receipt.total_device_bytes != cursor {
        return Err(ScoringArchiveArenaLayoutErrorV2::ReceiptTotalMismatch {
            expected: cursor,
            observed: receipt.total_device_bytes,
        });
    }
    if allocation.requested_bytes != cursor || allocation.aligned_bytes != cursor {
        return Err(ScoringArchiveArenaLayoutErrorV2::AllocationTotalMismatch {
            expected: cursor,
            observed_requested: allocation.requested_bytes,
            observed_aligned: allocation.aligned_bytes,
        });
    }

    Ok(ResidentScoringArchiveArenaLayoutV2 {
        fitness_scores,
        decision_keys,
        cub_scratch,
        archive_gene_scalars,
        archive_term_indices,
        archive_term_weights,
        archive_metric_rows,
        archive_signatures,
        archive_hashes,
        current_population_signatures,
        novelty_scores,
        exact_top_k_keys,
        admission_flags,
        admission_offsets,
        archive_control_and_seal,
        replacement_subtotal_bytes,
        total_device_bytes: cursor,
        stream_identity: allocation.stream_identity,
        pool_identity: allocation.pool_identity,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        ResidentSearchSlice2AlignedLayoutV2, ResidentSearchSlice2AllocationCategoryV2,
        ResidentSearchSlice2AsyncAllocationArgsV2, ResidentSearchSlice2ScoringArchiveReceiptV2,
    };
    use super::*;

    const HOST_FIXTURE_CUB_BYTES: u64 = 65_536;
    const HOST_FIXTURE_TOTAL_BYTES: u64 = 23_776_768;
    const ARCHIVE_ABI_SOURCE_V2: &str = include_str!("../native/resident_archive_knn_v2_abi.cuh");
    const ARCHIVE_CUDA_SOURCE_V2: &str = include_str!("../native/resident_archive_knn_v2.cu");
    const SEARCH_ABI_SOURCE_V2: &str =
        include_str!("../native/resident_search_generation_v2_abi.cuh");
    const POPULATION_CUDA_SOURCE_V2: &str = include_str!("../native/prototype_b_population.cu");
    const SCORING_CUDA_SOURCE_V2: &str = include_str!("../native/resident_scoring_novelty_v1.cu");
    const CUDA_BUILD_SOURCE_V2: &str = include_str!("../build.rs");

    #[test]
    fn slice2_composite_creator_receives_and_uses_the_frozen_bind_before_allocation() {
        for symbol in [
            "neoethos_gpu_cuda_population_query_resident_search_slice2_v3",
            "neoethos_gpu_cuda_population_create_resident_search_slice2_v3",
        ] {
            let call = format!("{symbol}(");
            assert_eq!(source_occurrences_v2(SEARCH_ABI_SOURCE_V2, &call), 1);
            assert_eq!(source_occurrences_v2(POPULATION_CUDA_SOURCE_V2, &call), 1);
        }

        let create = definition_body_v2(
            POPULATION_CUDA_SOURCE_V2,
            "neoethos_gpu_cuda_population_create_resident_search_slice2_v3",
        );
        assert!(create.contains("create_resident_search_combined_impl_v3("));
        assert!(create.contains("binding"));

        let combined = definition_body_v2(
            SCORING_CUDA_SOURCE_V2,
            "create_slice2_combined_scoring_archive_run_v2",
        );
        assert_eq!(source_occurrences_v2(combined, "cudaMallocAsync("), 1);
        assert!(!combined.contains("create_unbound_resident_scoring_run_v2("));
    }

    #[test]
    fn rust_archive_knn_v2_receipt_layouts_match_the_frozen_native_abi() {
        assert_eq!(std::mem::size_of::<RawResidentArchiveKnnPendingV2>(), 72);
        assert_eq!(std::mem::align_of::<RawResidentArchiveKnnPendingV2>(), 8);
        assert_eq!(
            std::mem::offset_of!(RawResidentArchiveKnnPendingV2, source_packed_commit_word),
            8
        );
        assert_eq!(
            std::mem::offset_of!(
                RawResidentArchiveKnnPendingV2,
                terminal_host_receipt_identity
            ),
            64
        );

        assert_eq!(std::mem::size_of::<RawResidentArchiveKnnTerminalV2>(), 104);
        assert_eq!(std::mem::align_of::<RawResidentArchiveKnnTerminalV2>(), 8);
        assert_eq!(
            std::mem::offset_of!(RawResidentArchiveKnnTerminalV2, receipt_identity),
            16
        );
        assert_eq!(
            std::mem::offset_of!(RawResidentArchiveKnnTerminalV2, validator_digest),
            96
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn rust_archive_knn_v2_ffi_signatures_match_the_frozen_native_abi() {
        use crate::population::RawResidentScoringPopulationSourceV2;
        use crate::resident_generation_v1::{NativeResidentGenerationRunV1, RawReadyEventV1};
        use crate::resident_search_v2::RawResidentGenerationGeneViewV2;
        use std::ffi::c_void;

        let _: unsafe extern "C" fn(
            *mut NativeResidentScoringNoveltyRunV1,
            *mut NativeResidentGenerationRunV1,
            *const RawResidentGenerationGeneViewV2,
            *const RawResidentArchiveKnnBindV2,
            *mut *mut NativeResidentArchiveKnnOwnerV2,
        ) -> i32 = bind_preallocated_resident_archive_knn_v2;
        let _: unsafe extern "C" fn(
            *mut NativeResidentArchiveKnnOwnerV2,
            *const RawResidentScoringPopulationSourceV2,
            *const RawReadyEventV1,
        ) -> i32 = enqueue_resident_archive_score_and_rank_v2;
        let _: unsafe extern "C" fn(*mut NativeResidentArchiveKnnOwnerV2) -> i32 =
            enqueue_resident_archive_stage_from_rank_v2;
        let _: unsafe extern "C" fn(*mut NativeResidentArchiveKnnOwnerV2) -> i32 =
            enqueue_resident_archive_evolve_and_publish_v2;
        let _: unsafe extern "C" fn(
            *mut NativeResidentArchiveKnnOwnerV2,
            *mut RawResidentArchiveKnnPendingV2,
        ) -> i32 = enqueue_resident_archive_terminal_seal_v2;
        let _: unsafe extern "C" fn(
            *mut NativeResidentArchiveKnnOwnerV2,
            *const RawResidentArchiveKnnPendingV2,
            *mut RawReadyEventV1,
            *mut RawResidentArchiveKnnTerminalV2,
        ) -> i32 = try_complete_resident_archive_terminal_v2;
        let _: unsafe extern "C" fn(*mut c_void, *mut NativeResidentArchiveKnnOwnerV2) -> i32 =
            neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2;
    }

    fn source_occurrences_v2(source: &str, needle: &str) -> usize {
        source.match_indices(needle).count()
    }

    fn definition_body_v2<'a>(source: &'a str, symbol: &str) -> &'a str {
        let needle = format!("{symbol}(");
        let symbol_offset = source
            .find(&needle)
            .unwrap_or_else(|| panic!("missing definition for `{symbol}`"));
        let body_offset = source[symbol_offset..]
            .find('{')
            .map(|offset| symbol_offset + offset)
            .unwrap_or_else(|| panic!("missing body for `{symbol}`"));
        let mut depth = 0_u64;
        for (relative_offset, byte) in source.as_bytes()[body_offset..].iter().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth = depth.checked_sub(1).expect("balanced definition braces");
                    if depth == 0 {
                        return &source[body_offset..=body_offset + relative_offset];
                    }
                }
                _ => {}
            }
        }
        panic!("unterminated body for `{symbol}`");
    }

    fn struct_body_v2<'a>(source: &'a str, name: &str) -> &'a str {
        let declaration = format!("struct {name}");
        let struct_offset = source
            .find(&declaration)
            .unwrap_or_else(|| panic!("missing definition for `{declaration}`"));
        let body_offset = source[struct_offset..]
            .find('{')
            .map(|offset| struct_offset + offset)
            .unwrap_or_else(|| panic!("missing body for `{declaration}`"));
        let mut depth = 0_u64;
        for (relative_offset, byte) in source.as_bytes()[body_offset..].iter().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth = depth.checked_sub(1).expect("balanced struct braces");
                    if depth == 0 {
                        return &source[body_offset..=body_offset + relative_offset];
                    }
                }
                _ => {}
            }
        }
        panic!("unterminated body for `{declaration}`");
    }

    fn compact_ascii_whitespace_v2(source: &str) -> String {
        source
            .split_ascii_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn remove_ascii_whitespace_v2(source: &str) -> String {
        source
            .chars()
            .filter(|character| !character.is_ascii_whitespace())
            .collect()
    }

    fn assert_source_excludes_v2(source: &str, forbidden: &[&str], scope: &str) {
        for token in forbidden {
            assert!(
                !source.contains(token),
                "{scope} must not contain forbidden token `{token}`"
            );
        }
    }

    fn validate_stable_three_pass_cub_rank_v2(source: &str) -> Result<(), String> {
        if source_occurrences_v2(source, "cub::DeviceRadixSort::SortPairs(") != 2 {
            return Err("rank must contain exactly two ascending stable CUB passes".to_owned());
        }
        if source_occurrences_v2(source, "cub::DeviceRadixSort::SortPairsDescending(") != 1 {
            return Err("rank must contain exactly one descending stable CUB pass".to_owned());
        }
        if source.contains("population_rank_less_v2")
            || source.contains("blend_and_rank_population_v2")
        {
            return Err("serial insertion ranking must be absent".to_owned());
        }

        let build =
            remove_ascii_whitespace_v2(definition_body_v2(source, "build_blended_rank_inputs_v2"));
        for seed in [
            "ordinal_keys[candidate]=candidate;",
            "ordinal_values[candidate]=candidate;",
        ] {
            if !build.contains(seed) {
                return Err(format!("ordinal-stability seed is missing `{seed}`"));
            }
        }
        if build.contains("while(") {
            return Err("rank input construction must not perform insertion sorting".to_owned());
        }

        let gene_gather = remove_ascii_whitespace_v2(definition_body_v2(
            source,
            "gather_gene_identity_rank_keys_v2",
        ));
        if !gene_gather.contains("gene_identity_keys[rank]=genes.scalars[ordinal].gene_identity;") {
            return Err("the second stable pass must gather gene identities".to_owned());
        }
        let blended_gather =
            remove_ascii_whitespace_v2(definition_body_v2(source, "gather_blended_rank_keys_v2"));
        if !blended_gather.contains("blended_keys[rank]=decision_keys[ordinal];") {
            return Err("the final stable pass must gather blended decision keys".to_owned());
        }

        let score = definition_body_v2(source, "enqueue_resident_archive_score_and_rank_v2");
        let compact_score = remove_ascii_whitespace_v2(score);
        for workspace in [
            "auto*rank_keys_a=owner->current_population_signatures;",
            "auto*rank_keys_b=rank_keys_a+owner->binding.population_count;",
            "auto*rank_values_a=rank_keys_b+owner->binding.population_count;",
            "auto*rank_values_b=rank_values_a+owner->binding.population_count;",
        ] {
            if !compact_score.contains(workspace) {
                return Err(format!("bounded rank workspace is missing `{workspace}`"));
            }
        }
        for preserved_output in [
            "reinterpret_cast<std::uint64_t*>(owner->exact_top_k_keys)",
            "reinterpret_cast<std::uint64_t*>(owner->novelty_scores)",
            "reinterpret_cast<std::uint64_t*>(owner->fitness_scores)",
        ] {
            if compact_score.contains(preserved_output) {
                return Err(format!(
                    "rank workspace must preserve authoritative output `{preserved_output}`"
                ));
            }
        }
        let copy =
            remove_ascii_whitespace_v2(definition_body_v2(source, "copy_ranked_ordinals_v2"));
        if !copy.contains("admission_offsets[rank]=ranked_ordinals[rank];") {
            return Err(
                "final CUB rank must be copied out before signatures are rebuilt".to_owned(),
            );
        }
        if source_occurrences_v2(score, "build_population_signatures_v2<<<") != 2 {
            return Err(
                "score/rank must rebuild signatures and admission flags after CUB reuse".to_owned(),
            );
        }
        for pass in [
            "cub::DeviceRadixSort::SortPairs(owner->cub_scratch,scratch_bytes,rank_keys_a,rank_keys_b,rank_values_a,rank_values_b,",
            "cub::DeviceRadixSort::SortPairs(owner->cub_scratch,scratch_bytes,rank_keys_a,rank_keys_b,rank_values_b,rank_values_a,",
            "cub::DeviceRadixSort::SortPairsDescending(owner->cub_scratch,scratch_bytes,rank_keys_a,rank_keys_b,rank_values_a,rank_values_b,",
        ] {
            if !compact_score.contains(pass) {
                return Err(format!("stable CUB pass is missing `{pass}`"));
            }
        }
        if source_occurrences_v2(score, "owner->binding.cub_scratch.size_bytes") != 3 {
            return Err("every CUB pass must reset the exact runtime scratch extent".to_owned());
        }

        let mut cursor = 0;
        for step in [
            "build_blended_rank_inputs_v2<<<",
            "cub::DeviceRadixSort::SortPairs(",
            "gather_gene_identity_rank_keys_v2<<<",
            "cub::DeviceRadixSort::SortPairs(",
            "gather_blended_rank_keys_v2<<<",
            "cub::DeviceRadixSort::SortPairsDescending(",
            "copy_ranked_ordinals_v2<<<",
            "build_population_signatures_v2<<<",
            "seal_ranked_population_v2<<<",
        ] {
            let relative = score[cursor..]
                .find(step)
                .ok_or_else(|| format!("rank chronology is missing `{step}`"))?;
            cursor += relative + step.len();
        }
        Ok(())
    }

    fn valid_receipt(cub_scratch_bytes: u64) -> ResidentSearchSlice2ScoringArchiveReceiptV2 {
        let layout = ResidentSearchSlice2AlignedLayoutV2 {
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
            replacement_subtotal_bytes: 23_707_648,
        };
        ResidentSearchSlice2ScoringArchiveReceiptV2 {
            fitness_score_bytes: 1_792,
            decision_key_bytes: 1_792,
            cub_scratch_bytes,
            total_device_bytes: 1_792 + 1_792 + cub_scratch_bytes + 23_707_648,
            layout,
        }
    }

    fn valid_allocation(total_device_bytes: u64) -> ResidentSearchSlice2AsyncAllocationArgsV2 {
        ResidentSearchSlice2AsyncAllocationArgsV2 {
            ordinal: 2,
            category: ResidentSearchSlice2AllocationCategoryV2::ScoringArchiveArena,
            requested_bytes: total_device_bytes,
            aligned_bytes: total_device_bytes,
            alignment_bytes: SCORING_ARCHIVE_ALIGNMENT_BYTES_V2,
            flags: 0,
            stream_identity: 0x1001,
            pool_identity: 0x2002,
        }
    }

    #[test]
    fn host_fixture_layout_has_every_exact_offset_and_end() {
        let receipt = valid_receipt(HOST_FIXTURE_CUB_BYTES);
        let allocation = valid_allocation(receipt.total_device_bytes);
        let authority =
            validate_scoring_archive_arena_layout_v2(allocation, receipt).expect("valid layout");

        assert_eq!(authority.fitness_scores, RegionV2::new(0, 1_792));
        assert_eq!(authority.decision_keys, RegionV2::new(1_792, 1_792));
        assert_eq!(authority.cub_scratch, RegionV2::new(3_584, 65_536));
        assert_eq!(
            authority.archive_gene_scalars,
            RegionV2::new(69_120, 3_600_128)
        );
        assert_eq!(
            authority.archive_term_indices,
            RegionV2::new(3_669_248, 6_400_000)
        );
        assert_eq!(
            authority.archive_term_weights,
            RegionV2::new(10_069_248, 6_400_000)
        );
        assert_eq!(
            authority.archive_metric_rows,
            RegionV2::new(16_469_248, 5_200_128)
        );
        assert_eq!(
            authority.archive_signatures,
            RegionV2::new(21_669_376, 1_600_000)
        );
        assert_eq!(authority.archive_hashes, RegionV2::new(23_269_376, 400_128));
        assert_eq!(
            authority.current_population_signatures,
            RegionV2::new(23_669_504, 6_400)
        );
        assert_eq!(authority.novelty_scores, RegionV2::new(23_675_904, 1_792));
        assert_eq!(
            authority.exact_top_k_keys,
            RegionV2::new(23_677_696, 96_000)
        );
        assert_eq!(authority.admission_flags, RegionV2::new(23_773_696, 1_024));
        assert_eq!(
            authority.admission_offsets,
            RegionV2::new(23_774_720, 1_792)
        );
        assert_eq!(
            authority.archive_control_and_seal,
            RegionV2::new(23_776_512, 256)
        );
        assert_eq!(authority.replacement_subtotal_bytes, 23_707_648);
        assert_eq!(authority.total_device_bytes, HOST_FIXTURE_TOTAL_BYTES);
        assert_eq!(authority.stream_identity, 0x1001);
        assert_eq!(authority.pool_identity, 0x2002);
    }

    #[test]
    fn runtime_cub_size_moves_following_offsets_with_checked_nonoverlap() {
        let receipt = valid_receipt(131_072);
        let allocation = valid_allocation(receipt.total_device_bytes);
        let authority =
            validate_scoring_archive_arena_layout_v2(allocation, receipt).expect("valid layout");
        let regions = authority.regions_v2();

        assert_eq!(authority.archive_gene_scalars.offset_bytes, 134_656);
        assert_eq!(authority.total_device_bytes, 23_842_304);
        for pair in regions.windows(2) {
            assert_eq!(pair[0].end_v2().expect("checked end"), pair[1].offset_bytes);
            assert_eq!(pair[1].offset_bytes % SCORING_ARCHIVE_ALIGNMENT_BYTES_V2, 0);
        }
        assert_eq!(
            regions.last().unwrap().end_v2(),
            Some(authority.total_device_bytes)
        );
    }

    #[test]
    fn one_field_and_total_drift_are_rejected() {
        let mut field_drift = valid_receipt(HOST_FIXTURE_CUB_BYTES);
        field_drift.layout.archive_hashes -= SCORING_ARCHIVE_ALIGNMENT_BYTES_V2;
        field_drift.layout.replacement_subtotal_bytes -= SCORING_ARCHIVE_ALIGNMENT_BYTES_V2;
        field_drift.total_device_bytes -= SCORING_ARCHIVE_ALIGNMENT_BYTES_V2;
        let field_allocation = valid_allocation(field_drift.total_device_bytes);
        assert_eq!(
            validate_scoring_archive_arena_layout_v2(field_allocation, field_drift),
            Err(ScoringArchiveArenaLayoutErrorV2::RegionSizeMismatch {
                region: ScoringArchiveArenaRegionV2::ArchiveHashes,
                expected: 400_128,
                observed: 399_872,
            })
        );

        let mut total_drift = valid_receipt(HOST_FIXTURE_CUB_BYTES);
        total_drift.total_device_bytes += SCORING_ARCHIVE_ALIGNMENT_BYTES_V2;
        let total_allocation = valid_allocation(total_drift.total_device_bytes);
        assert_eq!(
            validate_scoring_archive_arena_layout_v2(total_allocation, total_drift),
            Err(ScoringArchiveArenaLayoutErrorV2::ReceiptTotalMismatch {
                expected: HOST_FIXTURE_TOTAL_BYTES,
                observed: HOST_FIXTURE_TOTAL_BYTES + SCORING_ARCHIVE_ALIGNMENT_BYTES_V2,
            })
        );

        let mut overflow = valid_receipt(HOST_FIXTURE_CUB_BYTES);
        overflow.cub_scratch_bytes = u64::MAX - 255;
        overflow.total_device_bytes = 0;
        let overflow_allocation = valid_allocation(0);
        assert_eq!(
            validate_scoring_archive_arena_layout_v2(overflow_allocation, overflow),
            Err(ScoringArchiveArenaLayoutErrorV2::ArithmeticOverflow {
                region: ScoringArchiveArenaRegionV2::CubScratch,
            })
        );
    }

    #[test]
    fn allocation_contract_and_runtime_cub_alignment_drift_are_rejected() {
        let receipt = valid_receipt(HOST_FIXTURE_CUB_BYTES);

        let mut ordinal = valid_allocation(receipt.total_device_bytes);
        ordinal.ordinal = 1;
        assert_eq!(
            validate_scoring_archive_arena_layout_v2(ordinal, receipt),
            Err(ScoringArchiveArenaLayoutErrorV2::AllocationOrdinalMismatch { observed: 1 })
        );

        let mut category = valid_allocation(receipt.total_device_bytes);
        category.category = ResidentSearchSlice2AllocationCategoryV2::GenerationArena;
        assert_eq!(
            validate_scoring_archive_arena_layout_v2(category, receipt),
            Err(ScoringArchiveArenaLayoutErrorV2::AllocationCategoryMismatch)
        );

        let mut alignment = valid_allocation(receipt.total_device_bytes);
        alignment.alignment_bytes = 128;
        assert_eq!(
            validate_scoring_archive_arena_layout_v2(alignment, receipt),
            Err(ScoringArchiveArenaLayoutErrorV2::AllocationAlignmentMismatch { observed: 128 })
        );

        let mut flags = valid_allocation(receipt.total_device_bytes);
        flags.flags = 1;
        assert_eq!(
            validate_scoring_archive_arena_layout_v2(flags, receipt),
            Err(ScoringArchiveArenaLayoutErrorV2::AllocationFlagsMismatch { observed: 1 })
        );

        let unaligned_receipt = valid_receipt(HOST_FIXTURE_CUB_BYTES + 1);
        let unaligned_allocation = valid_allocation(unaligned_receipt.total_device_bytes);
        assert_eq!(
            validate_scoring_archive_arena_layout_v2(unaligned_allocation, unaligned_receipt),
            Err(
                ScoringArchiveArenaLayoutErrorV2::CubScratchAlignmentMismatch {
                    observed: HOST_FIXTURE_CUB_BYTES + 1,
                }
            )
        );
    }

    #[test]
    fn native_archive_knn_v2_declares_and_defines_only_the_seven_split_entrypoints() {
        const SPLIT_ENTRYPOINTS: [&str; 7] = [
            "bind_preallocated_resident_archive_knn_v2",
            "enqueue_resident_archive_score_and_rank_v2",
            "enqueue_resident_archive_stage_from_rank_v2",
            "enqueue_resident_archive_evolve_and_publish_v2",
            "enqueue_resident_archive_terminal_seal_v2",
            "try_complete_resident_archive_terminal_v2",
            "neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2",
        ];
        for symbol in SPLIT_ENTRYPOINTS {
            let call_token = format!("{symbol}(");
            assert_eq!(
                source_occurrences_v2(ARCHIVE_ABI_SOURCE_V2, &call_token),
                1,
                "ABI must declare `{symbol}` exactly once"
            );
            assert_eq!(
                source_occurrences_v2(ARCHIVE_CUDA_SOURCE_V2, &call_token),
                1,
                "CUDA TU must define `{symbol}` exactly once"
            );
        }
        assert_eq!(
            source_occurrences_v2(ARCHIVE_CUDA_SOURCE_V2, "extern \"C\""),
            SPLIT_ENTRYPOINTS.len(),
            "the dedicated CUDA TU must export exactly the seven C entrypoints"
        );

        for obsolete in [
            "query_resident_archive_knn_allocation_v2(",
            "query_resident_archive_knn_cub_scratch_v2(",
            "create_resident_archive_knn_owner_v2(",
            "enqueue_resident_archive_knn_generation_v2(",
            "try_complete_resident_archive_knn_generation_v2(",
        ] {
            assert_eq!(
                source_occurrences_v2(ARCHIVE_ABI_SOURCE_V2, obsolete)
                    + source_occurrences_v2(ARCHIVE_CUDA_SOURCE_V2, obsolete),
                0,
                "obsolete standalone/one-shot ABI `{obsolete}` must be removed"
            );
        }
    }

    #[test]
    fn native_archive_knn_v2_uses_stable_three_pass_cub_tuple_rank() {
        validate_stable_three_pass_cub_rank_v2(ARCHIVE_CUDA_SOURCE_V2)
            .expect("production rank must be stable three-pass CUB");

        let mutants = [
            ARCHIVE_CUDA_SOURCE_V2.replacen(
                "cub::DeviceRadixSort::SortPairsDescending(",
                "cub::DeviceRadixSort::SortPairs(",
                1,
            ),
            ARCHIVE_CUDA_SOURCE_V2.replacen(
                "gather_gene_identity_rank_keys_v2<<<",
                "gather_blended_rank_keys_v2<<<",
                1,
            ),
            ARCHIVE_CUDA_SOURCE_V2.replacen(
                "owner->cub_scratch, scratch_bytes",
                "nullptr, scratch_bytes",
                1,
            ),
            ARCHIVE_CUDA_SOURCE_V2.replacen(
                "ordinal_values[candidate] = candidate;",
                "ordinal_values[candidate] = candidate; while (position != 0) {}",
                1,
            ),
            ARCHIVE_CUDA_SOURCE_V2.replacen(
                "auto* rank_keys_a = owner->current_population_signatures;",
                "auto* rank_keys_a = reinterpret_cast<std::uint64_t*>(owner->exact_top_k_keys);",
                1,
            ),
            ARCHIVE_CUDA_SOURCE_V2.replacen(
                "auto* rank_keys_a = owner->current_population_signatures;",
                "auto* rank_keys_a = reinterpret_cast<std::uint64_t*>(owner->novelty_scores);",
                1,
            ),
        ];
        for mutant in mutants {
            assert_ne!(mutant, ARCHIVE_CUDA_SOURCE_V2, "mutant must alter source");
            assert!(
                validate_stable_three_pass_cub_rank_v2(&mutant).is_err(),
                "source contract must kill rank-order/scratch/insertion mutants"
            );
        }
    }

    #[test]
    fn native_archive_knn_v2_bind_dto_carries_the_dynamic_validated_arena_layout() {
        assert!(ARCHIVE_ABI_SOURCE_V2.contains("#include \"resident_generation_v2_abi.cuh\""));
        let region = struct_body_v2(ARCHIVE_ABI_SOURCE_V2, "NeoResidentArchiveKnnArenaRegionV2");
        assert_eq!(source_occurrences_v2(region, "offset_bytes"), 1);
        assert_eq!(source_occurrences_v2(region, "size_bytes"), 1);

        let bind = struct_body_v2(ARCHIVE_ABI_SOURCE_V2, "NeoResidentArchiveKnnBindV2");
        for region in [
            "fitness_scores",
            "decision_keys",
            "cub_scratch",
            "archive_gene_scalars",
            "archive_term_indices",
            "archive_term_weights",
            "archive_metric_rows",
            "archive_signatures",
            "archive_hashes",
            "current_population_signatures",
            "novelty_scores",
            "exact_top_k_keys",
            "admission_flags",
            "admission_offsets",
            "archive_control_and_seal",
        ] {
            assert_eq!(
                source_occurrences_v2(bind, region),
                1,
                "preallocated bind must carry nested region `{region}` exactly once"
            );
        }
        assert_eq!(
            source_occurrences_v2(bind, "NeoResidentArchiveKnnArenaRegionV2"),
            15,
            "preallocated bind must carry exactly fifteen typed regions"
        );
        for field in [
            "abi_version",
            "total_device_bytes",
            "device_uuid",
            "primary_context_identity",
            "search_stream_identity",
            "active_pool_identity",
            "cuda_build_identity",
            "kernel_semantics_identity",
            "binary64_math_identity",
            "plan_identity",
            "run_identity",
            "full_workspace_receipt_identity",
            "post_trim_receipt_identity",
            "population_count",
            "archive_capacity",
            "signature_word_count",
            "novelty_neighbor_count",
            "max_terms_per_gene",
        ] {
            assert_eq!(
                source_occurrences_v2(bind, field),
                1,
                "preallocated bind must carry identity/shape field `{field}` exactly once"
            );
        }
        assert_eq!(source_occurrences_v2(bind, "std::uint32_t reserved;"), 1);
        assert_eq!(
            source_occurrences_v2(bind, "std::uint32_t reserved_extents;"),
            1
        );
        assert!(
            compact_ascii_whitespace_v2(bind)
                .contains("std::uint32_t max_terms_per_gene; std::uint32_t reserved_extents;")
        );
        assert_source_excludes_v2(
            bind,
            &[
                "scoring_archive_arena_device",
                "cudaStream_t",
                "metric_count",
                "void *",
                "void*",
            ],
            "opaque preallocated bind",
        );

        let compact_header = compact_ascii_whitespace_v2(ARCHIVE_ABI_SOURCE_V2);
        assert!(compact_header.contains("sizeof(NeoResidentArchiveKnnArenaRegionV2) == 16"));
        assert!(compact_header.contains("sizeof(NeoResidentArchiveKnnBindV2) == 384"));
        assert!(compact_header.contains("sizeof(NeoResidentArchiveKnnPendingV2) == 72"));
        assert!(compact_header.contains("sizeof(NeoResidentArchiveKnnTerminalV2) == 104"));
        assert!(compact_header.contains("NEO_RESIDENT_ARCHIVE_KNN_METRIC_COUNT_V2 = 11"));
        assert!(ARCHIVE_ABI_SOURCE_V2.contains("NEO_ARCHIVE_KNN_TERMINAL_COMMITTED_V2"));
        assert!(ARCHIVE_ABI_SOURCE_V2.contains("NEO_ARCHIVE_KNN_TERMINAL_FAULT_V2"));
        assert!(!ARCHIVE_ABI_SOURCE_V2.contains("using NeoResidentArchiveKnnPendingV2"));
        assert!(!ARCHIVE_ABI_SOURCE_V2.contains("using NeoResidentArchiveKnnTerminalV2"));

        let pending = struct_body_v2(ARCHIVE_ABI_SOURCE_V2, "NeoResidentArchiveKnnPendingV2");
        for field in [
            "abi_version",
            "flags",
            "source_packed_commit_word",
            "terminal_device_receipt_identity",
            "run_identity",
            "boxed_receipt_identity",
            "staged_dependency_identity",
            "same_stream_enqueue_count",
            "completion_event_identity",
            "terminal_host_receipt_identity",
        ] {
            assert_eq!(
                source_occurrences_v2(pending, field),
                1,
                "private pending receipt must bind `{field}` exactly once"
            );
        }
        assert!(!pending.contains("target_packed_commit_word"));

        let terminal = struct_body_v2(ARCHIVE_ABI_SOURCE_V2, "NeoResidentArchiveKnnTerminalV2");
        for field in [
            "abi_version",
            "terminal_status",
            "device_fault_word",
            "validation_fault_word",
            "receipt_identity",
            "run_identity",
            "packed_commit_word",
            "collision_count",
            "compact_async_d2h_count",
            "compact_async_d2h_bytes",
            "completion_event_query_count",
            "completion_stream_synchronize_count",
            "same_stream_enqueue_count",
            "completion_event_identity",
            "validator_digest",
        ] {
            assert_eq!(
                source_occurrences_v2(terminal, field),
                1,
                "private terminal receipt must carry `{field}` exactly once"
            );
        }
        assert_source_excludes_v2(
            terminal,
            &[
                "current_store",
                "generation",
                "archive_count",
                "commit_epoch",
            ],
            "single-word terminal authority",
        );
        assert_source_excludes_v2(
            ARCHIVE_ABI_SOURCE_V2,
            &["65'536", "65536"],
            "dynamic archive ABI",
        );

        let compact_abi = remove_ascii_whitespace_v2(ARCHIVE_ABI_SOURCE_V2);
        let compact_cuda = remove_ascii_whitespace_v2(ARCHIVE_CUDA_SOURCE_V2);
        let bind_signature = "bind_preallocated_resident_archive_knn_v2(resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1*scoring,resident_generation_v1::NeoResidentGenerationRunV1*generation,constresident_generation_v2::NeoResidentGenerationGeneViewV2*genes,constNeoResidentArchiveKnnBindV2*binding,NeoResidentArchiveKnnOwnerV2**owner)";
        assert!(compact_abi.contains(bind_signature));
        assert!(compact_cuda.contains(bind_signature));
        let bind_definition = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "bind_preallocated_resident_archive_knn_v2",
        );
        assert!(bind_definition.contains("binding->reserved != 0"));
        assert!(bind_definition.contains("binding->reserved_extents != 0"));
        let release_signature = "neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2(void*session,NeoResidentArchiveKnnOwnerV2*owner)";
        assert!(compact_abi.contains(release_signature));
        assert!(compact_cuda.contains(release_signature));
    }

    #[test]
    fn native_archive_knn_v2_never_owns_allocations_frees_or_event_creation() {
        assert_source_excludes_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            &[
                "cudaMalloc",
                "cudaFree",
                "cudaHostAlloc",
                "cudaMemGetInfo",
                "cudaEventCreate",
            ],
            "borrowed archive CUDA TU",
        );
        let release = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2",
        );
        for required in [
            "HostPhaseV2::TerminalComplete",
            "terminal_event_proven",
            "population_lifetime_owner_v2()",
            "session !=",
        ] {
            assert!(
                release.contains(required),
                "borrowed archive release is missing `{required}`"
            );
        }
    }

    #[test]
    fn native_archive_knn_v2_split_transitions_stay_device_only_until_terminal() {
        for symbol in [
            "enqueue_resident_archive_score_and_rank_v2",
            "enqueue_resident_archive_stage_from_rank_v2",
            "enqueue_resident_archive_evolve_and_publish_v2",
        ] {
            let body = definition_body_v2(ARCHIVE_CUDA_SOURCE_V2, symbol);
            assert_source_excludes_v2(
                body,
                &[
                    "cudaMemcpyDeviceToHost",
                    "cudaMemcpy(",
                    "cudaEventRecord",
                    "cudaEventQuery",
                    "cudaStreamQuery",
                    "cudaEventSynchronize",
                    "cudaStreamSynchronize",
                    "cudaDeviceSynchronize",
                    "cuEventSynchronize",
                    "cuStreamSynchronize",
                ],
                symbol,
            );
        }

        let score = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "enqueue_resident_archive_score_and_rank_v2",
        );
        for required in [
            "owner->phase == HostPhaseV2::Bound",
            "owner->phase == HostPhaseV2::Published",
            "dependency == nullptr",
            "dependency != nullptr",
            "dependency != owner->terminal_lifecycle.source_ready_receipt_v2()",
            "dependency->event_id !=",
            "owner->terminal_lifecycle.source_event_id_v2()",
            "dependency->same_stream_enqueue_count",
            "owner->terminal_lifecycle.source_same_stream_enqueue_count_v2()",
            "population->metrics_ready_event !=",
            "owner->terminal_lifecycle.resident_parent_ready_event_v2()",
            "population->population_lifetime_owner !=",
            "owner->terminal_lifecycle.population_lifetime_owner_v2()",
            "finite_rows.same_stream_enqueue_count -",
            "advance_global_enqueue_count_v2(",
        ] {
            assert!(
                score.contains(required),
                "score phase is missing `{required}`"
            );
        }
        assert!(
            !score.contains(
                "finite_rows.same_stream_enqueue_count > owner->same_stream_enqueue_count"
            )
        );
        assert!(
            !score.contains("owner->same_stream_enqueue_count = dependency"),
            "the score phase must not replace the retained global count with caller data"
        );

        let evolve = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "enqueue_resident_archive_evolve_and_publish_v2",
        );
        assert_eq!(
            source_occurrences_v2(evolve, "borrow_resident_generation_terminal_lifecycle_v2("),
            2,
            "generation enqueue delta requires exact before/after snapshots"
        );
        assert!(evolve.contains("same_stream_enqueue_count_v2() -"));
        assert!(evolve.contains("owner->same_stream_enqueue_count +="));

        let terminal = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "enqueue_resident_archive_terminal_seal_v2",
        );
        assert_eq!(source_occurrences_v2(terminal, "cudaMemcpyAsync("), 1);
        assert_eq!(source_occurrences_v2(terminal, "cudaMemcpyDeviceToHost"), 1);
        assert_eq!(source_occurrences_v2(terminal, "cudaEventRecord("), 1);
        assert!(terminal.contains("sizeof(NeoResidentArchiveKnnTerminalV2)"));
        assert!(terminal.contains("owner->same_stream_enqueue_count + 3ull"));
        assert!(terminal.contains("lifecycle.same_stream_enqueue_count_v2() + 3ull"));
        assert_source_excludes_v2(
            terminal,
            &[
                "cudaMemcpy(",
                "cudaEventQuery",
                "cudaStreamQuery",
                "cudaEventSynchronize",
                "cudaStreamSynchronize",
                "cudaDeviceSynchronize",
            ],
            "terminal seal",
        );

        let poll = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "try_complete_resident_archive_terminal_v2",
        );
        assert_eq!(source_occurrences_v2(poll, "cudaEventQuery("), 1);
        assert_source_excludes_v2(
            poll,
            &[
                "cudaMemcpy",
                "cudaEventRecord",
                "cudaStreamQuery",
                "cudaEventSynchronize",
                "cudaStreamSynchronize",
                "cudaDeviceSynchronize",
                "cuEventSynchronize",
                "cuStreamSynchronize",
            ],
            "terminal poll",
        );
    }

    #[test]
    fn native_archive_knn_v2_collision_fallback_compares_the_full_normalized_gene() {
        let compact_header = remove_ascii_whitespace_v2(ARCHIVE_ABI_SOURCE_V2);
        let compact_cuda = remove_ascii_whitespace_v2(ARCHIVE_CUDA_SOURCE_V2);
        assert!(!compact_header.contains("exact_gene[2]"));
        assert!(!compact_cuda.contains("exact_gene[2]"));
        assert!(ARCHIVE_ABI_SOURCE_V2.contains("NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2 = 16"));
        assert!(ARCHIVE_CUDA_SOURCE_V2.contains("NeoResidentGenerationGeneScalarV1"));
        assert!(ARCHIVE_CUDA_SOURCE_V2.contains("NeoResidentGenerationGeneViewV2"));

        let equality =
            definition_body_v2(ARCHIVE_CUDA_SOURCE_V2, "full_fixed_stride_gene_equal_v2");
        for scalar_field in [
            "term_count",
            "smc_flags",
            "long_threshold",
            "short_threshold",
            "target_pips",
            "stop_pips",
            "stop_vol_multiplier",
        ] {
            assert!(
                source_occurrences_v2(equality, scalar_field) >= 2,
                "full equality must compare both `{scalar_field}` values"
            );
        }
        for token in [
            "NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2",
            "term_indices",
            "term_weights",
            "f64_bits",
        ] {
            assert!(
                equality.contains(token),
                "full equality is missing `{token}`"
            );
        }
        assert!(
            source_occurrences_v2(equality, "f64_bits") >= 12,
            "five scalar f64 fields and all term weights require bitwise equality"
        );
        assert!(source_occurrences_v2(equality, "term_indices") >= 2);
        assert!(source_occurrences_v2(equality, "term_weights") >= 2);
        assert!(
            compact_ascii_whitespace_v2(equality)
                .contains("< NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2")
        );
    }

    #[test]
    fn native_archive_knn_v2_translation_unit_and_header_are_registered_once() {
        assert_eq!(
            source_occurrences_v2(
                CUDA_BUILD_SOURCE_V2,
                "\"native/resident_archive_knn_v2.cu\""
            ),
            1,
            "archive CUDA TU must appear exactly once in build.rs"
        );
        assert_eq!(
            source_occurrences_v2(
                CUDA_BUILD_SOURCE_V2,
                "\"native/resident_archive_knn_v2_abi.cuh\""
            ),
            1,
            "archive ABI header must appear exactly once in build.rs"
        );
    }
}
