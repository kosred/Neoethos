use super::{
    ResidentSearchSlice2AllocationCategoryV2, ResidentSearchSlice2AsyncAllocationArgsV2,
    ResidentSearchSlice2ScoringArchiveReceiptV2,
};

const SCORING_ARCHIVE_ALIGNMENT_BYTES_V2: u64 = 256;
const FITNESS_SCORE_BYTES_V2: u64 = 1_792;
const DECISION_KEY_BYTES_V2: u64 = 1_792;
const REPLACEMENT_SUBTOTAL_BYTES_V2: u64 = 23_707_648;

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
}
