use super::ResidentSearchSlice2CalibrationBindingV2;
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::population::RawResidentScoringPopulationSourceV2;
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::resident_archive_output_v3::{
    RawResidentArchiveExportReceiptV3, RawResidentArchiveGeneScalarV3,
    ResidentArchiveExportContextV3,
};
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::resident_generation_v1::{
    NativeResidentGenerationRunV1, RawReadyEventV1, selected_generation_abi_v1,
};
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::resident_search_v2::RawResidentGenerationGeneViewV2;
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use std::ffi::c_void;

/// Select the native archive protocol, not a device capability or a caller ID.
/// CUDA keeps its existing wire value; the HIP build uses its distinct ABI.
pub(crate) const fn selected_archive_abi_v2() -> u32 {
    if cfg!(feature = "hip-native-kernels") {
        0x0001_0002
    } else {
        2
    }
}

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
    #[cfg(not(feature = "hip-native-kernels"))]
    pub(super) reserved: u32,
    #[cfg(feature = "hip-native-kernels")]
    pub(super) backend_kind: u32,
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
    #[cfg(not(feature = "hip-native-kernels"))]
    pub(super) primary_context_identity: u64,
    #[cfg(feature = "hip-native-kernels")]
    pub(super) hip_lease_identity: u64,
    pub(super) search_stream_identity: u64,
    pub(super) active_pool_identity: u64,
    #[cfg(not(feature = "hip-native-kernels"))]
    pub(super) cuda_build_identity: u64,
    #[cfg(feature = "hip-native-kernels")]
    pub(super) hip_build_identity: u64,
    pub(super) kernel_semantics_identity: u64,
    pub(super) binary64_math_identity: u64,
    pub(super) plan_identity: u64,
    pub(super) run_identity: u64,
    pub(super) full_workspace_receipt_identity: u64,
    pub(super) post_trim_receipt_identity: u64,
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
pub(crate) enum NativeResidentScoringNoveltyRunV1 {}
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
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

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
impl RawResidentArchiveKnnTerminalV2 {
    pub(crate) fn archive_export_context_v3(
        &self,
        feature_count: u64,
        max_terms: u32,
    ) -> ResidentArchiveExportContextV3 {
        ResidentArchiveExportContextV3 {
            run_identity: self.run_identity,
            packed_commit_word: self.packed_commit_word,
            candidate_count: (self.packed_commit_word >> 17) & 0xffff,
            feature_count,
            max_terms,
            terminal_generation: (self.packed_commit_word >> 1) & 0xffff,
        }
    }

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
        binding.selected_backend_matches_v2()
            && pending.abi_version == selected_archive_abi_v2()
            && pending.flags == 0
            && self.abi_version == selected_archive_abi_v2()
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
            && ready.abi_version == selected_generation_abi_v1()
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

impl RawResidentArchiveKnnBindV2 {
    pub(crate) fn archive_capacity_v3(&self) -> u64 {
        self.archive_capacity
    }

    #[cfg(any(feature = "cuda", feature = "hip-native-kernels", test))]
    fn selected_backend_matches_v2(&self) -> bool {
        #[cfg(feature = "hip-native-kernels")]
        let backend_matches = self.backend_kind == 2;
        #[cfg(not(feature = "hip-native-kernels"))]
        let backend_matches = self.reserved == 0;
        self.abi_version == selected_archive_abi_v2() && backend_matches
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
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

    pub(crate) fn copy_resident_archive_terminal_candidates_v4(
        owner: *mut NativeResidentArchiveKnnOwnerV2,
        expected_terminal: *const RawResidentArchiveKnnTerminalV2,
        scalars: *mut RawResidentArchiveGeneScalarV3,
        term_indices: *mut u64,
        term_weights: *mut f64,
        metrics: *mut neoethos_gpu_contracts::device::NeoPopulationMetricRow,
        admission_sequences: *mut u64,
        candidate_capacity: u64,
        term_capacity: u64,
        receipt: *mut RawResidentArchiveExportReceiptV3,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegionV2 {
    offset_bytes: u64,
    size_bytes: u64,
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
    total_device_bytes: u64,
}

impl ResidentScoringArchiveArenaLayoutV2 {
    fn archive_retention_word_count_v3(archive_capacity: u64) -> Result<u64, &'static str> {
        if archive_capacity == 0 {
            return Err("Slice2 archive retention requires nonzero capacity");
        }
        let table_capacity = archive_capacity
            .checked_mul(2)
            .and_then(u64::checked_next_power_of_two)
            .ok_or("Slice2 archive hash table capacity overflow")?;
        archive_capacity
            .checked_mul(6)
            .and_then(|words| words.checked_add(table_capacity))
            .ok_or("Slice2 archive retention word count overflow")
    }

    /// Exact frozen ABI layout. Only the CUB extent is device/toolkit dependent;
    /// it must come from the native preliminary combined-admission query.
    pub(super) fn from_native_scratch_v2(
        population_count: u64,
        archive_capacity: u64,
        cub_scratch_bytes: u64,
        signature_word_count: u32,
        novelty_neighbor_count: u32,
    ) -> Result<Self, &'static str> {
        if population_count == 0 || population_count > i32::MAX as u64 {
            return Err("Slice2 population exceeds the native CUB item-count domain");
        }
        if archive_capacity == 0 || archive_capacity > u16::MAX as u64 {
            return Err("Slice2 archive capacity exceeds the native packed-count domain");
        }
        if cub_scratch_bytes == 0 || cub_scratch_bytes % 256 != 0 {
            return Err("Slice2 requires nonzero aligned native CUB scratch measurement");
        }
        if signature_word_count < 4 {
            return Err("Slice2 signatures must also fit four disjoint CUB key/value arrays");
        }
        if novelty_neighbor_count == 0 {
            return Err("Slice2 requires a nonzero configured novelty neighborhood");
        }
        fn region(cursor: &mut u64, count: u64, stride: u64) -> Result<RegionV2, &'static str> {
            let bytes = count
                .checked_mul(stride)
                .ok_or("Slice2 region product overflow")?;
            let size_bytes = bytes
                .checked_add(255)
                .ok_or("Slice2 region alignment overflow")?
                / 256
                * 256;
            let offset_bytes = *cursor;
            *cursor = cursor
                .checked_add(size_bytes)
                .ok_or("Slice2 arena offset overflow")?;
            Ok(RegionV2 {
                offset_bytes,
                size_bytes,
            })
        }
        let mut cursor = 0;
        // Native sizeof(GeneScalarV1)=72, metric row=104 including candidate
        // and scenario identities, exact kNN key=32.
        // At least four signature words provide the four population-sized u64
        // arrays reused by stable CUB ranking before ALL words are rebuilt.
        let fitness_scores = region(&mut cursor, population_count, 8)?;
        let decision_keys = region(&mut cursor, population_count, 8)?;
        let cub_scratch = region(&mut cursor, 1, cub_scratch_bytes)?;
        // Active and staged banks coexist until the generation commits. A
        // failed replacement must leave every prior archive byte recoverable.
        let archive_bank_count = archive_capacity.checked_mul(2).ok_or("Slice2 archive banks overflow")?;
        let archive_gene_scalars = region(&mut cursor, archive_bank_count, 72)?;
        let archive_term_indices = region(&mut cursor, archive_bank_count, 16 * 8)?;
        let archive_term_weights = region(&mut cursor, archive_bank_count, 16 * 8)?;
        let archive_metric_rows = region(
            &mut cursor,
            archive_bank_count,
            std::mem::size_of::<neoethos_gpu_contracts::device::NeoPopulationMetricRow>() as u64,
        )?;
        let signature_stride_bytes = u64::from(signature_word_count) * 8;
        let archive_signatures = region(&mut cursor, archive_bank_count, signature_stride_bytes)?;
        // Keep the two content-hash and two admission-sequence banks first.
        // The remaining words hold a <=50%-loaded slot+1 hash table, an indexed
        // min-heap and its inverse, rebuilt from the staged bank each generation.
        let archive_hashes = region(
            &mut cursor,
            Self::archive_retention_word_count_v3(archive_capacity)?,
            8,
        )?;
        let current_population_signatures =
            region(&mut cursor, population_count, signature_stride_bytes)?;
        let novelty_scores = region(&mut cursor, population_count, 8)?;
        let exact_top_k_keys = region(&mut cursor, population_count, u64::from(novelty_neighbor_count) * 32)?;
        let admission_flags = region(&mut cursor, population_count, 4)?;
        let admission_offsets = region(&mut cursor, population_count, 8)?;
        let archive_control_and_seal = region(&mut cursor, 1, 256)?;
        if cursor > isize::MAX as u64 {
            return Err("Slice2 arena exceeds the native addressable allocation domain");
        }
        Ok(Self {
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
            total_device_bytes: cursor,
        })
    }

    pub(super) fn total_device_bytes_v2(&self) -> u64 {
        self.total_device_bytes
    }

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
            abi_version: selected_archive_abi_v2(),
            #[cfg(not(feature = "hip-native-kernels"))]
            reserved: 0,
            #[cfg(feature = "hip-native-kernels")]
            backend_kind: 2,
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
            #[cfg(not(feature = "hip-native-kernels"))]
            primary_context_identity: calibration.primary_context_identity,
            #[cfg(feature = "hip-native-kernels")]
            hip_lease_identity: calibration.hip_lease_identity,
            search_stream_identity: calibration.search_stream_identity,
            active_pool_identity: calibration.active_pool_identity,
            #[cfg(not(feature = "hip-native-kernels"))]
            cuda_build_identity: calibration.cuda_build_identity,
            #[cfg(feature = "hip-native-kernels")]
            hip_build_identity: calibration.hip_build_identity,
            kernel_semantics_identity: calibration.kernel_semantics_identity,
            binary64_math_identity: calibration.binary64_math_identity,
            plan_identity: calibration.plan_identity,
            run_identity: calibration.run_identity,
            full_workspace_receipt_identity,
            post_trim_receipt_identity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic wire values exercise the pure layout/validation code only;
    // they never enter native code or manufacture a runtime admission owner.
    fn host_test_binding_v3() -> RawResidentArchiveKnnBindV2 {
        ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(7, 9, 256, 4, 3)
            .unwrap()
            .into_native_bind_v2(
                ResidentSearchSlice2CalibrationBindingV2 {
                    device_uuid: [11; 16],
                    #[cfg(not(feature = "hip-native-kernels"))]
                    primary_context_identity: 13,
                    #[cfg(feature = "hip-native-kernels")]
                    hip_lease_identity: 13,
                    search_stream_identity: 17,
                    active_pool_identity: 19,
                    #[cfg(not(feature = "hip-native-kernels"))]
                    cuda_build_identity: 23,
                    #[cfg(feature = "hip-native-kernels")]
                    hip_build_identity: 23,
                    kernel_semantics_identity: 29,
                    binary64_math_identity: 31,
                    plan_identity: 37,
                    run_identity: 41,
                },
                7,
                9,
                4,
                3,
                4,
                43,
                47,
            )
    }

    #[test]
    fn archive_selected_backend_keeps_exact_wire_offsets_and_calibration() {
        let binding = host_test_binding_v3();
        assert_eq!(std::mem::size_of::<RawResidentArchiveKnnBindV2>(), 384);
        assert_eq!(std::mem::align_of::<RawResidentArchiveKnnBindV2>(), 8);
        assert_eq!(
            std::mem::offset_of!(RawResidentArchiveKnnBindV2, device_uuid),
            288
        );
        assert_eq!(
            std::mem::offset_of!(RawResidentArchiveKnnBindV2, search_stream_identity),
            312
        );
        assert_eq!(
            std::mem::offset_of!(RawResidentArchiveKnnBindV2, active_pool_identity),
            320
        );
        #[cfg(not(feature = "hip-native-kernels"))]
        {
            assert_eq!(selected_archive_abi_v2(), 2);
            assert_eq!(
                std::mem::offset_of!(RawResidentArchiveKnnBindV2, reserved),
                4
            );
            assert_eq!(
                std::mem::offset_of!(RawResidentArchiveKnnBindV2, primary_context_identity),
                304
            );
            assert_eq!(
                std::mem::offset_of!(RawResidentArchiveKnnBindV2, cuda_build_identity),
                328
            );
            assert_eq!(
                (
                    binding.reserved,
                    binding.primary_context_identity,
                    binding.cuda_build_identity
                ),
                (0, 13, 23)
            );
        }
        #[cfg(feature = "hip-native-kernels")]
        {
            assert_eq!(selected_archive_abi_v2(), 0x0001_0002);
            assert_eq!(
                std::mem::offset_of!(RawResidentArchiveKnnBindV2, backend_kind),
                4
            );
            assert_eq!(
                std::mem::offset_of!(RawResidentArchiveKnnBindV2, hip_lease_identity),
                304
            );
            assert_eq!(
                std::mem::offset_of!(RawResidentArchiveKnnBindV2, hip_build_identity),
                328
            );
            assert_eq!(
                (
                    binding.backend_kind,
                    binding.hip_lease_identity,
                    binding.hip_build_identity
                ),
                (2, 13, 23)
            );
        }
        assert_eq!(binding.device_uuid, [11; 16]);
        assert_eq!(
            (binding.search_stream_identity, binding.active_pool_identity),
            (17, 19)
        );
        assert_eq!(
            (
                binding.kernel_semantics_identity,
                binding.binary64_math_identity
            ),
            (29, 31)
        );
        assert_eq!((binding.plan_identity, binding.run_identity), (37, 41));
        assert_eq!(
            (
                binding.full_workspace_receipt_identity,
                binding.post_trim_receipt_identity
            ),
            (43, 47)
        );
        assert_eq!(
            (
                binding.signature_word_count,
                binding.novelty_neighbor_count,
                binding.max_terms_per_gene
            ),
            (4, 3, 4)
        );
        assert!(binding.selected_backend_matches_v2());
        let mut foreign = binding;
        foreign.abi_version ^= 0x0001_0000;
        assert!(!foreign.selected_backend_matches_v2());
        let mut malformed = binding;
        #[cfg(feature = "hip-native-kernels")]
        {
            malformed.backend_kind = 0;
        }
        #[cfg(not(feature = "hip-native-kernels"))]
        {
            malformed.reserved = 2;
        }
        assert!(!malformed.selected_backend_matches_v2());
    }

    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    #[test]
    fn archive_terminal_requires_matching_backend_pending_and_ready_protocols() {
        let binding = host_test_binding_v3();
        let pending = RawResidentArchiveKnnPendingV2 {
            abi_version: selected_archive_abi_v2(),
            run_identity: 41,
            terminal_host_receipt_identity: 67,
            completion_event_identity: 73,
            same_stream_enqueue_count: 79,
            ..Default::default()
        };
        let ready = RawReadyEventV1 {
            abi_version: selected_generation_abi_v1(),
            event_id: 73,
            generation_index: 3,
            same_stream_enqueue_count: 79,
            ..Default::default()
        };
        let terminal = RawResidentArchiveKnnTerminalV2 {
            abi_version: selected_archive_abi_v2(),
            terminal_status: 1,
            receipt_identity: 67,
            run_identity: 41,
            packed_commit_word: 0x0000_0002_0004_0007, // store1, generation3, count2, epoch1
            collision_count: 7,
            compact_async_d2h_count: 1,
            compact_async_d2h_bytes: 104,
            completion_event_query_count: 1,
            same_stream_enqueue_count: 79,
            completion_event_identity: 73,
            // Literal little-endian FNV-1a checkpoint for [packed, 7, 41, 0].
            validator_digest: 0xab89_9d2f_f8b5_49cc,
            ..Default::default()
        };
        assert!(terminal.validates_committed_v2(&pending, &binding, &ready));
        let mut wrong_binding = binding;
        wrong_binding.abi_version ^= 0x0001_0000;
        assert!(!terminal.validates_committed_v2(&pending, &wrong_binding, &ready));
        let mut wrong_pending = pending;
        wrong_pending.abi_version ^= 0x0001_0000;
        assert!(!terminal.validates_committed_v2(&wrong_pending, &binding, &ready));
        wrong_pending = pending;
        wrong_pending.flags = 1;
        assert!(!terminal.validates_committed_v2(&wrong_pending, &binding, &ready));
        let mut wrong_terminal = terminal;
        wrong_terminal.abi_version ^= 0x0001_0000;
        assert!(!wrong_terminal.validates_committed_v2(&pending, &binding, &ready));
        let mut wrong_ready = ready;
        wrong_ready.abi_version ^= 0x0001_0000;
        assert!(!terminal.validates_committed_v2(&pending, &binding, &wrong_ready));
        // Keep existing event, transfer and fault checks active on both backends.
        for mutate in [
            |v: &mut RawResidentArchiveKnnTerminalV2| v.device_fault_word = 1,
            |v: &mut RawResidentArchiveKnnTerminalV2| v.validation_fault_word = 1,
            |v: &mut RawResidentArchiveKnnTerminalV2| v.compact_async_d2h_count = 2,
            |v: &mut RawResidentArchiveKnnTerminalV2| v.compact_async_d2h_bytes = 105,
            |v: &mut RawResidentArchiveKnnTerminalV2| v.completion_event_query_count = 0,
            |v: &mut RawResidentArchiveKnnTerminalV2| v.completion_stream_synchronize_count = 1,
            |v: &mut RawResidentArchiveKnnTerminalV2| v.completion_event_identity ^= 1,
            |v: &mut RawResidentArchiveKnnTerminalV2| v.same_stream_enqueue_count ^= 1,
            |v: &mut RawResidentArchiveKnnTerminalV2| v.validator_digest ^= 1,
        ] {
            let mut invalid = terminal;
            mutate(&mut invalid);
            assert!(!invalid.validates_committed_v2(&pending, &binding, &ready));
        }
    }

    #[test]
    fn slice2_native_admission_layout_matches_all_fifteen_native_regions() {
        let population = 33;
        let archive = 65;
        let scratch = 1_024;
        let layout = ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
            population, archive, scratch, 4, 15,
        )
        .expect("checked layout, not a device measurement");
        let regions = [
            (layout.fitness_scores, population * 8),
            (layout.decision_keys, population * 8),
            (layout.cub_scratch, scratch),
            (layout.archive_gene_scalars, 2 * archive * 72),
            (layout.archive_term_indices, 2 * archive * 16 * 8),
            (layout.archive_term_weights, 2 * archive * 16 * 8),
            (layout.archive_metric_rows, 2 * archive * 104),
            (layout.archive_signatures, 2 * archive * 4 * 8),
            (layout.archive_hashes, (6 * archive + 256) * 8),
            (layout.current_population_signatures, population * 4 * 8),
            (layout.novelty_scores, population * 8),
            (layout.exact_top_k_keys, population * 15 * 32),
            (layout.admission_flags, population * 4),
            (layout.admission_offsets, population * 8),
            (layout.archive_control_and_seal, 256),
        ];
        let mut end = 0;
        for (region, logical_bytes) in regions {
            assert_eq!(region.offset_bytes, end);
            assert_eq!(region.size_bytes, logical_bytes.div_ceil(256) * 256);
            assert_eq!(region.offset_bytes % 256, 0);
            end += region.size_bytes;
        }
        assert_eq!(layout.total_device_bytes_v2(), end);
        type MetricRow = neoethos_gpu_contracts::device::NeoPopulationMetricRow;
        assert_eq!(std::mem::size_of::<MetricRow>(), 104);
        assert_eq!(std::mem::offset_of!(MetricRow, candidate_id), 0);
        assert_eq!(std::mem::offset_of!(MetricRow, scenario_id), 8);
        assert_eq!(std::mem::offset_of!(MetricRow, values), 16);
        let scoring_abi = include_str!("../native/resident_scoring_novelty_v1_abi.cuh");
        assert!(
            scoring_abi
                .contains("using NeoResidentScoringNoveltyMetricRowV1 = ::NeoPopulationMetricRow;")
        );
        assert!(scoring_abi.contains("sizeof(NeoResidentScoringNoveltyMetricRowV1) == 104"));
        assert!(layout.current_population_signatures.size_bytes >= population * 4 * 8);
        assert!(ARCHIVE_CUDA_SOURCE_V2.contains("static_assert(sizeof(ExactNeighborKeyV2) == 32"));
        assert!(
            include_str!("../native/resident_generation_v1_abi.cuh")
                .contains("sizeof(NeoResidentGenerationGeneScalarV1) == 72")
        );
    }

    #[test]
    fn slice2_native_admission_layout_refuses_unmeasured_scratch_and_overflow() {
        for (population, archive, scratch) in [
            (0, 1, 256),
            (1, 0, 256),
            (i32::MAX as u64 + 1, 1, 256),
            (1, u16::MAX as u64 + 1, 256),
            (1, 1, 0),
            (1, 1, 255),
            (1, 1, u64::MAX - 255),
        ] {
            assert!(
                ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
                    population, archive, scratch, 4, 15,
                )
                .is_err()
            );
        }
        for words in [0, 1, 3] {
            assert!(
                ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(1, 1, 256, words, 15)
                    .is_err()
            );
        }
        assert!(
            ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
                i32::MAX as u64,
                1,
                256,
                u32::MAX,
                15,
            )
            .is_err()
        );
    }

    #[test]
    fn slice2_native_admission_layout_charges_wide_signatures_without_raising_term_stride() {
        let population = 200;
        let archive = 65;
        let narrow = ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
            population, archive, 1_024, 4, 15,
        )
        .unwrap();
        let wide = ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
            population, archive, 1_024, 31, 15,
        )
        .unwrap();
        let align = |bytes: u64| bytes.div_ceil(256) * 256;
        assert_eq!(wide.archive_signatures.size_bytes, align(2 * archive * 31 * 8));
        assert_eq!(
            wide.current_population_signatures.size_bytes,
            align(population * 31 * 8)
        );
        assert_eq!(
            wide.total_device_bytes_v2() - narrow.total_device_bytes_v2(),
            align(2 * archive * 31 * 8) - align(2 * archive * 4 * 8) + align(population * 31 * 8)
                - align(population * 4 * 8)
        );
        assert_eq!(wide.archive_term_indices, narrow.archive_term_indices);
        assert_eq!(wide.archive_term_weights, narrow.archive_term_weights);
        assert_eq!(wide.cub_scratch, narrow.cub_scratch);
        assert!(wide.current_population_signatures.size_bytes >= population * 4 * 8);
    }

    #[test]
    fn slice2_native_admission_layout_charges_configured_neighbors_and_rollback_banks() {
        let align = |bytes: u64| bytes.div_ceil(256) * 256;
        let population = 33;
        let capacity = 65;
        for neighbors in [1_u32, 7, 15, 31, 65] {
            let layout = ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
                population, capacity, 1_024, 4, neighbors,
            ).unwrap();
            assert_eq!(layout.exact_top_k_keys.size_bytes, align(population * u64::from(neighbors) * 32));
            assert_eq!(layout.archive_gene_scalars.size_bytes, align(2 * capacity * 72));
            assert_eq!(layout.archive_term_indices.size_bytes, align(2 * capacity * 16 * 8));
            assert_eq!(layout.archive_term_weights.size_bytes, align(2 * capacity * 16 * 8));
            assert_eq!(layout.archive_metric_rows.size_bytes, align(2 * capacity * 104));
            assert_eq!(layout.archive_signatures.size_bytes, align(2 * capacity * 4 * 8));
            assert_eq!(layout.archive_hashes.size_bytes, align((6 * capacity + 256) * 8));
            // The old one-bank reservation cannot fit the inactive payload;
            // this negative control must fail even after alignment padding.
            assert!(layout.archive_metric_rows.size_bytes > align(capacity * 104));
        }
        assert!(ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
            1, 1, 256, 4, 0,
        ).is_err());
        assert!(ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
            i32::MAX as u64, 1, 256, 4, u32::MAX,
        ).is_err());
    }

    #[test]
    fn slice2_native_admission_layout_charges_indexed_retention_at_capacity_boundaries() {
        let align = |bytes: u64| bytes.div_ceil(256) * 256;
        for (capacity, table_capacity) in [
            (1_u64, 2_u64),
            (2, 4),
            (3, 8),
            (4, 8),
            (5, 16),
            (7, 16),
            (8, 16),
            (9, 32),
            (32_767, 65_536),
            (32_768, 65_536),
            (32_769, 131_072),
            (65_535, 131_072),
        ] {
            let layout = ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
                3, capacity, 256, 4, 15,
            ).unwrap();
            let table_start = 4 * capacity;
            let heap_start = table_start + table_capacity;
            let inverse_start = heap_start + capacity;
            let words = inverse_start + capacity;
            assert_eq!(
                ResidentScoringArchiveArenaLayoutV2::archive_retention_word_count_v3(capacity),
                Ok(words),
            );
            assert!(table_capacity.is_power_of_two());
            assert!(table_capacity >= 2 * capacity);
            assert_eq!(layout.archive_hashes.size_bytes, align(words * 8));
            assert!(layout.archive_hashes.size_bytes >= (inverse_start + capacity) * 8);
            assert_eq!(
                layout.current_population_signatures.offset_bytes,
                layout.archive_hashes.offset_bytes + layout.archive_hashes.size_bytes,
            );
            // The old four-bank-only logical extent never covers the index.
            // For tiny capacities existing alignment padding can cover it;
            // once A>=5 the old physical reservation is also insufficient.
            assert!(words * 8 > 4 * capacity * 8);
            if capacity >= 5 {
                assert!(layout.archive_hashes.size_bytes > align(4 * capacity * 8));
            }
        }
    }

    #[test]
    fn slice2_native_admission_retention_words_refuse_zero_and_checked_overflow() {
        for capacity in [
            0,
            u64::MAX,                 // 2A overflows.
            (1_u64 << 62) + 1,         // The next power of two overflows.
            u64::MAX / 6 + 1,          // 6A overflows.
            1_u64 << 61,              // 6A and H fit separately, but their sum does not.
        ] {
            assert!(
                ResidentScoringArchiveArenaLayoutV2::archive_retention_word_count_v3(capacity)
                    .is_err(),
                "capacity={capacity}",
            );
        }
        // The public layout keeps the existing packed archive-count bounds.
        for capacity in [0, 65_536, u64::MAX] {
            assert!(ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
                3, capacity, 256, 4, 15,
            ).is_err());
        }
    }

    #[test]
    fn native_archive_knn_v2_dynamic_signatures_preserve_cub_rebuild_and_active_stride() {
        let build = definition_body_v2(ARCHIVE_CUDA_SOURCE_V2, "build_population_signatures_v2");
        assert_source_steps_v2(
            build,
            &[
                "scalar.term_count > expected.max_terms_per_gene",
                "signatures + candidate * signature_word_count",
                "word < signature_word_count",
                "signature[word] = 0ull;",
                "candidate * expected.max_terms_per_gene",
                "term < expected.max_terms_per_gene",
                "feature >= expected.feature_count",
                "feature / 64ull >= signature_word_count",
                "signature[feature / 64ull] |= 1ull << (feature % 64ull);",
            ],
        );
        assert!(!build.contains("local[NEO_RESIDENT_ARCHIVE_KNN_SIGNATURE_WORDS_V2]"));
        validate_stable_three_pass_cub_rank_v2(ARCHIVE_CUDA_SOURCE_V2).unwrap();
        let rank = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "enqueue_resident_archive_score_and_rank_v2",
        );
        assert_eq!(
            source_occurrences_v2(rank, "owner->binding.signature_word_count"),
            3
        );
        let knn = definition_body_v2(ARCHIVE_CUDA_SOURCE_V2, "exact_archive_population_knn_v2");
        assert_source_steps_v2(
            knn,
            &[
                "std::uint64_t intersection = 0;",
                "std::uint64_t union_count = 0;",
                "word < signature_word_count",
                "union_count > 32",
                "static_cast<std::uint32_t>(union_count - intersection)",
            ],
        );
        let stage = definition_body_v2(ARCHIVE_CUDA_SOURCE_V2, "stage_ranked_archive_tail_v2");
        assert_source_steps_v2(
            stage,
            &[
                "current.term_weights, candidate, expected.max_terms_per_gene",
                "archive_term_weights, archived, NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2",
                "term < NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2",
                "term < expected.max_terms_per_gene ? current.term_indices[candidate * expected.max_terms_per_gene + term] : 0ull",
                "term < expected.max_terms_per_gene ? current.term_weights[candidate * expected.max_terms_per_gene + term] : 0.0",
                "word < signature_word_count",
                "archive_signatures[destination * signature_word_count + word]",
                "current_signatures[candidate * signature_word_count + word]",
            ],
        );
        let scoring = include_str!("../native/resident_scoring_novelty_v1.cu");
        for name in [
            "query_slice2_combined_scoring_archive_run_v2",
            "create_slice2_combined_scoring_archive_run_v2",
        ] {
            let body = definition_body_v2(scoring, name);
            assert!(remove_ascii_whitespace_v2(body).contains(
                "binding->signature_word_count!=resident_archive_knn_v2::signature_word_count_v2(plan->feature_count)"
            ));
            assert!(body.contains("binding->max_terms_per_gene != plan->max_terms_per_gene"));
        }
    }

    #[test]
    fn native_archive_knn_v2_economic_rejection_keeps_signatures_and_does_not_mask_faults() {
        let build = definition_body_v2(ARCHIVE_CUDA_SOURCE_V2, "build_population_signatures_v2");
        assert_source_steps_v2(
            build,
            &[
                "scoring_seal_valid_v2(scoring_seal)",
                "scalar.term_count > expected.max_terms_per_gene",
                "classify_resident_metrics_v2(row.values)",
                "metric_status == ResidentMetricStatusV2::Fault",
                "latch_device_fault_v2(control, kNonFiniteMetricFaultV2)",
                "signature[word] = 0ull",
                "signature[feature / 64ull] |= 1ull << (feature % 64ull)",
                "const bool mode_passed = policy.mode == 1u",
                "row.values[5] > policy.minimum_profit_factor",
                "row.values[1] > policy.minimum_sharpe",
                "row.values[kNetMetricSlotV2] > policy.minimum_net",
                "metric_status == ResidentMetricStatusV2::Finite && row.values[kTradeCountMetricSlotV2] > 0.0 && mode_passed",
            ],
        );
        // An invalid monthly-equity metric is excluded, but a finite-metric
        // objective rejection must not silently change CPU archive eligibility.
        assert!(!build.contains("fitness_scores"));
        let blend = definition_body_v2(ARCHIVE_CUDA_SOURCE_V2, "build_blended_rank_inputs_v2");
        assert_source_steps_v2(
            blend,
            &[
                "scoring_seal_valid_v2(scoring_seal)",
                "!isfinite(fitness_scores[candidate]) && fitness_scores[candidate] != rejected_fitness",
                "!isfinite(novelty_scores[candidate])",
                "latch_device_fault_v2(control, kNonFiniteMetricFaultV2)",
                "if (isfinite(fitness_scores[candidate]))",
                "minimum_fitness =",
                "maximum_fitness =",
                "maximum_novelty =",
                "double fitness_range = any_finite_fitness ? __dsub_rn(maximum_fitness, minimum_fitness) : 1.0e-9",
                "ordinal_keys[candidate] = candidate",
                "ordinal_values[candidate] = candidate",
                "fitness_scores[candidate] == rejected_fitness",
                "decision_keys[candidate] = 1ull",
                "continue;",
                "const double normalized_fitness =",
                "decision_keys[candidate] = ordered_finite_f64_key_v2(blended)",
            ],
        );
        validate_stable_three_pass_cub_rank_v2(ARCHIVE_CUDA_SOURCE_V2).unwrap();
    }

    const ARCHIVE_ABI_SOURCE_V2: &str = include_str!("../native/resident_archive_knn_v2_abi.cuh");
    const ARCHIVE_CUDA_SOURCE_V2: &str = include_str!("../native/resident_archive_knn_v2.cu");
    fn production_archive_source_v2() -> String {
        const GUARD: &str = "#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)";
        assert_eq!(ARCHIVE_CUDA_SOURCE_V2.matches(GUARD).count(), 1);
        let start = ARCHIVE_CUDA_SOURCE_V2.find(GUARD).unwrap();
        let end = start + ARCHIVE_CUDA_SOURCE_V2[start..].find("#endif").unwrap() + "#endif".len();
        let fixture = &ARCHIVE_CUDA_SOURCE_V2[start..end];
        assert_eq!(fixture.matches("#if").count(), 1, "fixture guard cannot nest or hide production");
        assert_eq!(fixture.matches("extern \"C\"").count(), 1);
        assert!(fixture.contains("extern \"C\" std::int32_t fixture_check_adaptive_archive_index_v3("));
        assert!(fixture.contains("fixture_check_adaptive_archive_index_kernel_v3<<<1, 1"));
        assert_eq!(ARCHIVE_CUDA_SOURCE_V2[end..].trim(), "}  // namespace neoethos::resident_archive_knn_v2");
        // Exempt only this named final, feature-gated algorithm fixture. Every
        // production function still participates in the original whole-TU bans.
        format!("{}{}", &ARCHIVE_CUDA_SOURCE_V2[..start], &ARCHIVE_CUDA_SOURCE_V2[end..])
    }
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
            "neoethos_gpu_cuda_population_create_resident_search_slice2_adaptive_v3",
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

    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
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
        let _: unsafe extern "C" fn(
            *mut NativeResidentArchiveKnnOwnerV2,
            *const RawResidentArchiveKnnTerminalV2,
            *mut RawResidentArchiveGeneScalarV3,
            *mut u64,
            *mut f64,
            *mut neoethos_gpu_contracts::device::NeoPopulationMetricRow,
            *mut u64,
            u64,
            u64,
            *mut RawResidentArchiveExportReceiptV3,
        ) -> i32 = copy_resident_archive_terminal_candidates_v4;
    }

    fn source_occurrences_v2(source: &str, needle: &str) -> usize {
        source.match_indices(needle).count()
    }

    fn source_simple_assignments_v3(source: &str, target: &str) -> usize {
        source
            .match_indices(target)
            .filter(|(offset, _)| {
                let suffix = source[offset + target.len()..].trim_start();
                suffix.starts_with('=') && !suffix.starts_with("==")
            })
            .count()
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

    fn assert_source_steps_v2(source: &str, steps: &[&str]) {
        let source = remove_ascii_whitespace_v2(source);
        let mut cursor = 0;
        for step in steps {
            let step = remove_ascii_whitespace_v2(step);
            let offset = source[cursor..]
                .find(&step)
                .unwrap_or_else(|| panic!("missing or out-of-order source step `{step}`"));
            cursor += offset + step.len();
        }
    }

    fn validate_stable_three_pass_cub_rank_v2(source: &str) -> Result<(), String> {
        if source_occurrences_v2(
            source,
            "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairs(",
        ) != 2
        {
            return Err("rank must contain exactly two ascending stable CUB passes".to_owned());
        }
        if source_occurrences_v2(
            source,
            "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairsDescending(",
        ) != 1
        {
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
            "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairs(owner->cub_scratch,scratch_bytes,rank_keys_a,rank_keys_b,rank_values_a,rank_values_b,",
            "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairs(owner->cub_scratch,scratch_bytes,rank_keys_a,rank_keys_b,rank_values_b,rank_values_a,",
            "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairsDescending(owner->cub_scratch,scratch_bytes,rank_keys_a,rank_keys_b,rank_values_a,rank_values_b,",
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
            "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairs(",
            "gather_gene_identity_rank_keys_v2<<<",
            "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairs(",
            "gather_blended_rank_keys_v2<<<",
            "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairsDescending(",
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

    #[test]
    fn native_archive_knn_v2_declares_ten_lifecycle_policy_and_terminal_export_entrypoints() {
        const SPLIT_ENTRYPOINTS: [&str; 10] = [
            "bind_preallocated_resident_archive_knn_v2",
            "configure_resident_archive_novelty_v3",
            "configure_resident_archive_policy_v3",
            "enqueue_resident_archive_score_and_rank_v2",
            "enqueue_resident_archive_stage_from_rank_v2",
            "enqueue_resident_archive_evolve_and_publish_v2",
            "enqueue_resident_archive_terminal_seal_v2",
            "try_complete_resident_archive_terminal_v2",
            "neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2",
            "copy_resident_archive_terminal_candidates_v4",
        ];
        for symbol in SPLIT_ENTRYPOINTS {
            let call_token = format!("{symbol}(");
            assert_eq!(
                source_occurrences_v2(ARCHIVE_ABI_SOURCE_V2, &call_token),
                1,
                "ABI must declare `{symbol}` exactly once"
            );
            assert_eq!(
                source_occurrences_v2(&remove_ascii_whitespace_v2(ARCHIVE_CUDA_SOURCE_V2),
                    &remove_ascii_whitespace_v2(&format!("extern \"C\" std::int32_t {symbol}("))),
                1,
                "CUDA TU must define `{symbol}` exactly once"
            );
        }
        assert_eq!(
            source_occurrences_v2(&production_archive_source_v2(), "extern \"C\""),
            SPLIT_ENTRYPOINTS.len(),
            "the dedicated CUDA TU exposes seven lifecycle entries, two immutable policy setters and terminal-only export"
        );

        let configure = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "configure_resident_archive_novelty_v3",
        );
        assert_source_steps_v2(
            configure,
            &[
                "owner == nullptr || !std::isfinite(novelty_weight)",
                "novelty_weight < 0.0 || novelty_weight > 1.0",
                "return NEO_ARCHIVE_KNN_STATUS_INVALID_ARGUMENT_V2;",
                "owner->poisoned || owner->phase != HostPhaseV2::Bound || owner->novelty_configured",
                "return NEO_ARCHIVE_KNN_STATUS_STATE_ERROR_V2;",
                "expected_run_identity == 0 || expected_run_identity != owner->binding.run_identity",
                "expected_run_identity != owner->retained_gene_view.expected_run_token",
                "return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;",
                "owner->novelty_weight = novelty_weight;",
                "owner->novelty_configured = true;",
                "return NEO_ARCHIVE_KNN_STATUS_OK_V2;",
            ],
        );
        assert_eq!(
            source_simple_assignments_v3(ARCHIVE_CUDA_SOURCE_V2, "owner->novelty_weight"),
            1,
            "only the checked, pre-generation setter may change the run policy"
        );
        // A comparison is not a write, but an extra assignment with ANY RHS
        // must still fail the policy-immutability guard.
        let changed_policy = format!("{ARCHIVE_CUDA_SOURCE_V2}\nowner->novelty_weight = 0.35;");
        assert_eq!(
            source_simple_assignments_v3(&changed_policy, "owner->novelty_weight"),
            2
        );
        let score = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "enqueue_resident_archive_score_and_rank_v2",
        );
        assert!(score.contains("owner->novelty_weight"));
        assert_source_excludes_v2(
            configure,
            &["cudaMemcpy", "cudaMalloc", "cudaEvent", "<<<"],
            "immutable pre-generation policy setter",
        );
        let policy = definition_body_v2(ARCHIVE_CUDA_SOURCE_V2, "configure_resident_archive_policy_v3");
        assert_source_steps_v2(policy, &[
            "policy == nullptr || policy->abi_version != 3u || policy->mode > 3u",
            "!std::isfinite(policy->minimum_net)", "!std::isfinite(policy->minimum_profit_factor)",
            "!std::isfinite(policy->minimum_sharpe)", "configure_resident_archive_novelty_v3(",
            "if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2) return status;",
            "owner->policy = *policy", "owner->adaptive_policy_configured = true",
        ]);
        assert_source_excludes_v2(policy, &["cudaMemcpy", "cudaMalloc", "cudaEvent", "<<<"],
            "immutable pre-generation adaptive archive policy setter");

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
                "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairsDescending(",
                "neoethos_parallel_primitives_v1::DeviceRadixSort::SortPairs(",
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
            "hip_lease_identity",
            "search_stream_identity",
            "active_pool_identity",
            "cuda_build_identity",
            "hip_build_identity",
            "backend_kind",
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
        // Power-of-two hash-table boundary assertions legitimately mention
        // 65,536 slots; that is not a fixed archive capacity or payload array.
        let header_without_index_boundary_assertions = ARCHIVE_ABI_SOURCE_V2.lines()
            .filter(|line| !line.trim_start().starts_with("static_assert(archive_hash_table_capacity_v3("))
            .collect::<Vec<_>>().join("\n");
        assert_source_excludes_v2(
            &header_without_index_boundary_assertions,
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
        assert!(bind_definition.contains("!backend_identity_v3::archive_backend_valid(*binding)"));
        assert!(bind_definition.contains("binding->reserved_extents != 0"));
        let release_signature = "neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2(void*session,NeoResidentArchiveKnnOwnerV2*owner)";
        assert!(compact_abi.contains(release_signature));
        assert!(compact_cuda.contains(release_signature));
    }

    fn validate_archive_backend_source_v3(source: &str) -> Result<(), &'static str> {
        for (symbol, hip, cuda) in [
            (
                "archive_backend_valid",
                "value.backend_kind==2u",
                "value.reserved==0u",
            ),
            (
                "archive_owner_identity",
                "(value.hip_lease_identity)",
                "(value.primary_context_identity)",
            ),
            (
                "archive_build_identity",
                "(value.hip_build_identity)",
                "(value.cuda_build_identity)",
            ),
        ] {
            let body = remove_ascii_whitespace_v2(definition_body_v2(source, symbol));
            let branches =
                format!("#ifdefined(__HIP_PLATFORM_AMD__)return{hip};#elsereturn{cuda};#endif");
            if !body.contains(&branches) {
                return Err("archive selector must retain both exact backend identities");
            }
        }
        Ok(())
    }

    #[test]
    fn native_archive_backend_selectors_keep_cuda_checks_and_real_hip_owner_checks() {
        let source = include_str!("../native/resident_backend_identity_v3.cuh");
        validate_archive_backend_source_v3(source).unwrap();
        for (old, replacement) in [
            ("value.backend_kind == 2u", "value.backend_kind == 0u"),
            ("value.reserved == 0u", "true"),
            (
                "(value.hip_lease_identity)",
                "(value.primary_context_identity)",
            ),
            ("(value.hip_build_identity)", "(value.cuda_build_identity)"),
        ] {
            let mutant = source.replacen(old, replacement, 1);
            assert_ne!(mutant, source);
            assert!(validate_archive_backend_source_v3(&mutant).is_err());
        }
        assert!(ARCHIVE_ABI_SOURCE_V2.contains("NEO_RESIDENT_ARCHIVE_KNN_ABI_V2 = 0x00010002u"));
        assert!(ARCHIVE_ABI_SOURCE_V2.contains("NEO_RESIDENT_ARCHIVE_KNN_ABI_V2 = 2;"));
        let export = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "copy_resident_archive_terminal_candidates_v4",
        );
        assert_source_steps_v2(
            export,
            &[
                "std::memcmp(&normalized, owner->terminal_host, sizeof(normalized)) != 0",
                "#if defined(__HIP_PLATFORM_AMD__)",
                "!resident_search_hip_v1::validate_population_owner_v1(",
                "owner->terminal_lifecycle.population_lifetime_owner_v2()",
                "owner->binding, owner->arena_access.admitted_run_stream",
                "owner->poisoned = true;",
                "return NEO_ARCHIVE_KNN_STATUS_IDENTITY_MISMATCH_V2;",
                "#else",
                "cuCtxGetCurrent(&current_context) != CUDA_SUCCESS",
                "cuCtxGetId(current_context, &current_context_id) != CUDA_SUCCESS",
                "current_context_id != backend_identity_v3::archive_owner_identity(owner->binding)",
                "#endif",
                "count > owner->binding.archive_capacity",
                "cudaMemcpy(scalars,",
            ],
        );
        // These are the same native sorter implementations used for both the
        // scratch query and execution, not a replacement host ranking path.
        let primitives = include_str!("../native/resident_parallel_primitives_v1.cuh");
        assert_source_steps_v2(
            primitives,
            &[
                "#if defined(__HIP_PLATFORM_AMD__)",
                "#include <hipcub/hipcub.hpp>",
                "namespace neoethos_parallel_primitives_v1 = ::hipcub;",
                "#else",
                "#include <cub/cub.cuh>",
                "namespace neoethos_parallel_primitives_v1 = ::cub;",
                "#endif",
            ],
        );
    }

    #[test]
    fn native_archive_knn_v2_never_owns_allocations_frees_or_event_creation() {
        assert_source_excludes_v2(
            &production_archive_source_v2(),
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
    fn native_archive_knn_v2_publish_updates_the_retained_alias_before_reuse() {
        let generation = include_str!("../native/resident_generation_v1.cu");
        let evolve = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "enqueue_resident_archive_evolve_and_publish_v2",
        );
        // The prepared pointer aliases this exact owner member; the existing
        // accept function updates it. A second archive-side setter is not needed.
        assert_source_steps_v2(
            evolve,
            &[
                "enqueue_resident_generation_offspring_from_finite_rows_v2(owner->generation, &owner->finite_rows, owner->decision_keys, &owner->retained_gene_view, &prepared)",
                "publish_generation_and_archive_v2<<<",
                "status = launch_status_v2();",
                "if (status != NEO_ARCHIVE_KNN_STATUS_OK_V2)",
                "accept_resident_generation_combined_publish_v2(&prepared)",
                "if (status != 0)",
                "TerminalLifecycleV2 generation_after{};",
                "owner->phase = HostPhaseV2::Published;",
            ],
        );
        let offspring = definition_body_v2(
            generation,
            "enqueue_resident_generation_offspring_from_finite_rows_v2",
        );
        assert!(
            remove_ascii_whitespace_v2(offspring)
                .contains("prepared->retained_generation_view_=retained_generation_view;")
        );
        let accept =
            definition_body_v2(generation, "accept_resident_generation_combined_publish_v2");
        assert_source_steps_v2(
            accept,
            &[
                "prepared->retained_generation_view_ != nullptr",
                "prepared->retained_generation_view_->expected_generation_index == prepared->expected_old_generation_index_",
                "prepared->retained_generation_view_->expected_store_epoch == prepared->expected_old_store_epoch_",
                "if (!exact)",
                "return NEO_RESIDENT_STATUS_IDENTITY_MISMATCH_V1;",
                "rotate_resident_generation_stores_v1(generation);",
                "prepared->retained_generation_view_->expected_generation_index = prepared->expected_next_generation_index_;",
                "prepared->retained_generation_view_->expected_store_epoch = prepared->expected_next_store_epoch_;",
                "generation->ready_receipt_token_v2 = nullptr;",
                "return NEO_RESIDENT_STATUS_OK_V1;",
            ],
        );
        assert_source_excludes_v2(
            accept,
            &[
                "cudaMemcpy",
                "cudaEventSynchronize",
                "cudaStreamSynchronize",
            ],
            "planned host bookkeeping is not device completion proof",
        );
    }

    #[test]
    fn native_archive_knn_v2_terminal_export_checks_identity_extents_and_copy_completion() {
        let export = definition_body_v2(
            ARCHIVE_CUDA_SOURCE_V2,
            "copy_resident_archive_terminal_candidates_v4",
        );
        assert_source_steps_v2(
            export,
            &[
                "*receipt = {};",
                "owner == nullptr || expected_terminal == nullptr",
                "owner->poisoned || owner->phase != HostPhaseV2::TerminalComplete",
                "!owner->terminal_event_proven || owner->terminal_host == nullptr",
                "owner->candidates_exported",
                "normalized.completion_event_query_count != owner->completion_event_query_count",
                "normalized.terminal_status != NEO_ARCHIVE_KNN_TERMINAL_COMMITTED_V2",
                "normalized.device_fault_word != 0",
                "normalized.validation_fault_word != 0",
                "normalized.completion_event_query_count = 0;",
                "std::memcmp(&normalized, owner->terminal_host, sizeof(normalized)) != 0",
                "cuCtxGetCurrent(&current_context) != CUDA_SUCCESS",
                "cuCtxGetId(current_context, &current_context_id) != CUDA_SUCCESS",
                "current_context_id != backend_identity_v3::archive_owner_identity(owner->binding)",
                "count > owner->binding.archive_capacity",
                "!checked_mul_v2(count, NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2, &terms)",
                "!checked_mul_v2(count, sizeof(GeneScalarV2), &scalar_bytes)",
                "!checked_mul_v2(terms, sizeof(std::uint64_t), &index_bytes)",
                "!checked_mul_v2(terms, sizeof(double), &weight_bytes)",
                "!checked_mul_v2(count, sizeof(MetricRowV2), &metric_bytes)",
                "!checked_mul_v2(count, sizeof(std::uint64_t), &sequence_bytes)",
                "!checked_add_v2(scalar_bytes, index_bytes, &total_bytes)",
                "!checked_add_v2(total_bytes, weight_bytes, &total_bytes)",
                "!checked_add_v2(total_bytes, metric_bytes, &total_bytes)",
                "!checked_add_v2(total_bytes, sequence_bytes, &total_bytes)",
                "candidate_capacity != count || term_capacity != terms",
                "count != 0 && (scalars == nullptr || term_indices == nullptr || term_weights == nullptr || metrics == nullptr || admission_sequences == nullptr)",
                "const auto bank_offset = owner->adaptive_policy_configured",
                "unpack_store_v2(normalized.packed_commit_word) * owner->binding.archive_capacity : 0ull",
                "const auto bank_terms = bank_offset * NEO_RESIDENT_ARCHIVE_KNN_MAX_TERMS_V2",
                "if (count != 0 &&",
                "cudaMemcpy(scalars, owner->archive_gene_scalars + bank_offset, scalar_bytes, cudaMemcpyDeviceToHost) != cudaSuccess",
                "cudaMemcpy(term_indices, owner->archive_term_indices + bank_terms, index_bytes, cudaMemcpyDeviceToHost) != cudaSuccess",
                "cudaMemcpy(term_weights, owner->archive_term_weights + bank_terms, weight_bytes, cudaMemcpyDeviceToHost) != cudaSuccess",
                "cudaMemcpy(metrics, owner->archive_metric_rows + bank_offset, metric_bytes, cudaMemcpyDeviceToHost) != cudaSuccess",
                "cudaMemcpy(admission_sequences, owner->archive_hashes + 2 * owner->binding.archive_capacity + bank_offset, sequence_bytes, cudaMemcpyDeviceToHost) != cudaSuccess",
                "return poison_owner_v2(owner, NEO_ARCHIVE_KNN_STATUS_CUDA_ERROR_V2);",
                "receipt->abi_version = 4;",
                "receipt->run_identity = normalized.run_identity;",
                "receipt->packed_commit_word = normalized.packed_commit_word;",
                "receipt->candidate_count = count;",
                "receipt->term_count = terms;",
                "receipt->feature_count = owner->retained_gene_view.feature_count;",
                "receipt->host_copy_count = count == 0 ? 0 : 5;",
                "receipt->host_copy_bytes = total_bytes;",
                "owner->candidates_exported = true;",
                "return NEO_ARCHIVE_KNN_STATUS_OK_V2;",
            ],
        );
        assert_eq!(source_occurrences_v2(export, "cudaMemcpy("), 5);
        assert_source_excludes_v2(
            export,
            &["cudaMemcpyAsync", "cudaMalloc", "cudaEventRecord", "<<<"],
            "one-shot completed archive export",
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
