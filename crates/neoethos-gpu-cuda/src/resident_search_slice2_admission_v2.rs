//! Run-bound Slice2 admission built after the native preliminary sizing query.
//! CUB scratch and free memory are observed on the admitted device; archive
//! extents are checked derivations. This is not a performance calibration.

#[path = "resident_archive_knn_v2_native.rs"]
pub(crate) mod resident_archive_knn_v2_native;

use self::resident_archive_knn_v2_native::{
    RawResidentArchiveKnnBindV2, ResidentScoringArchiveArenaLayoutV2,
};
#[cfg(feature = "cuda")]
use crate::resident_feature_store_v3::ResidentPopulationSessionV3;
use crate::resident_generation_v1::SealedResidentGenerationPlanV1;
use crate::resident_scoring_v2::{
    SealedResidentSearchAdmissionV2, seal_combined_search_admission_v2,
};

/// Actual input carrier, not a synthetic trim receipt for compact Data.
pub(crate) enum ResidentSearchSlice2InputIdentityV3 {
    #[cfg(feature = "cuda")]
    Compact(ResidentSearchCompactInputIdentityV3),
    #[cfg(feature = "hip-native-kernels")]
    Hip {
        parent: crate::hip_runtime_v1::feature_store_v1::HipSearchParentIdentityV3,
        requested_plan_sha256: [u8; 32],
        requested_scope_sha256: [u8; 32],
    },
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResidentSearchCompactInputIdentityV3 {
    admission_sha256: [u8; 32],
    content_merkle: [u8; 32],
    retirement_token: [u8; 32],
    device: neoethos_gpu_contracts::resident_feature_store_v3::CudaPrimaryContextBuildIdentityV3,
    limits: crate::data_population_workspace_plan_v1::SealedDataPopulationExecutionLimitsV1,
    rows: u64,
    columns: u64,
}

#[cfg(feature = "cuda")]
impl ResidentSearchCompactInputIdentityV3 {
    pub(crate) fn from_session_v3(
        session: &ResidentPopulationSessionV3,
    ) -> Result<Self, &'static str> {
        let identity = Self {
            admission_sha256: session.admission_identity_sha256(),
            content_merkle: session.canonical_content_merkle(),
            retirement_token: session.data_transient_retirement_process_token(),
            device: session.device_identity().clone(),
            limits: *session
                .data_population_limits()
                .ok_or("compact Search requires sealed Data+population limits")?,
            rows: u64::try_from(session.rows()).map_err(|_| "compact row extent exceeds u64")?,
            columns: u64::try_from(session.columns())
                .map_err(|_| "compact column extent exceeds u64")?,
        };
        if identity.admission_sha256 == [0; 32]
            || identity.content_merkle == [0; 32]
            || identity.retirement_token == [0; 32]
            || identity.limits.workspace_plan_identity_sha256() == [0; 32]
            || identity.limits.population_sizing_authority_sha256() == [0; 32]
            || identity.rows == 0
            || identity.columns == 0
            || identity.rows != identity.limits.parent_row_count()
            || identity.columns != identity.limits.feature_count()
        {
            return Err("compact Search input lost its Data/session authority");
        }
        Ok(identity)
    }

    pub(crate) fn matches_session_v3(
        &self,
        session: &ResidentPopulationSessionV3,
    ) -> Result<(), &'static str> {
        if self != &Self::from_session_v3(session)? {
            return Err("compact Search execution plan belongs to a different Data/session owner");
        }
        Ok(())
    }
}

struct ResidentSearchInputBindingsV3 {
    admission_sha256: [u8; 32],
    selection_sha256: [u8; 32],
    math_sha256: [u8; 32],
    ordinal: u32,
    device_uuid: Option<[u8; 16]>,
    #[cfg(feature = "hip-native-kernels")]
    hip_runtime: crate::hip_runtime_v1::HipRuntimeIdentityV1,
}

impl ResidentSearchSlice2InputIdentityV3 {
    fn bind_plan_v3(
        &self,
        plan: &SealedResidentGenerationPlanV1,
    ) -> Result<ResidentSearchInputBindingsV3, &'static str> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Compact(identity) => {
                let terms = plan
                    .logical_population_count_v1()
                    .checked_mul(u64::from(plan.max_terms_per_gene_v1()))
                    .ok_or("compact Search term extent overflow")?;
                if plan.feature_count_v1() != identity.columns
                    || plan.logical_population_count_v1() != identity.limits.max_candidate_count()
                    || terms > identity.limits.max_gene_term_count()
                    || plan.cuda_build_manifest_sha256_v1()
                        != identity.device.gpu_cuda_build_sha256()
                {
                    return Err(
                        "Slice2 generation plan differs from the exact compact Data/population authority",
                    );
                }
                Ok(ResidentSearchInputBindingsV3 {
                    admission_sha256: identity.admission_sha256,
                    // The existing opaque native selection slot carries the real
                    // compact sizing identity, never a manufactured trim receipt.
                    selection_sha256: identity.limits.population_sizing_authority_sha256(),
                    math_sha256: crate::resident_scoring_v2::cuda_math_flags_sha256_v2(),
                    ordinal: identity.device.ordinal(),
                    device_uuid: Some(identity.device.device_uuid()),
                })
            }
            #[cfg(feature = "hip-native-kernels")]
            Self::Hip {
                parent,
                requested_plan_sha256,
                requested_scope_sha256,
            } => {
                if plan.plan_identity_sha256_v1() != *requested_plan_sha256
                    || plan.feature_count_v1() != parent.features as u64
                    || plan.native_build_manifest_sha256_v1() != parent.native_build_sha256
                    || parent.physical_binding_sha256 == [0; 32]
                    || parent.content_merkle == [0; 32]
                    || *requested_scope_sha256 == [0; 32]
                {
                    return Err("HIP generation request differs from the retained physical parent");
                }
                Ok(ResidentSearchInputBindingsV3 {
                    admission_sha256: parent.physical_binding_sha256,
                    // This opaque native slot binds the actual requested scope,
                    // not a fabricated trim or financial selection receipt.
                    selection_sha256: *requested_scope_sha256,
                    math_sha256: crate::resident_scoring_v2::native_math_flags_sha256_v3(),
                    ordinal: parent.runtime.device_ordinal(),
                    device_uuid: Some(parent.runtime.device_uuid()),
                    hip_runtime: parent.runtime.clone(),
                })
            }
        }
    }

    pub(crate) fn validate_plan_v3(
        &self,
        plan: &SealedResidentGenerationPlanV1,
    ) -> Result<(), &'static str> {
        self.bind_plan_v3(plan).map(|_| ())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResidentSearchSlice2MeasuredAdmissionV2 {
    preliminary_receipt_sha256: [u8; 32],
    import_admission_sha256: [u8; 32],
    post_trim_plan_sha256: [u8; 32],
    same_context_free_bytes: u64,
    allocator_context_reserve_bytes: u64,
    generation_device_bytes: u64,
    scoring_archive_device_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2CalibrationBindingV2 {
    pub(crate) device_uuid: [u8; 16],
    #[cfg(not(feature = "hip-native-kernels"))]
    pub(crate) primary_context_identity: u64,
    #[cfg(feature = "hip-native-kernels")]
    pub(crate) hip_lease_identity: u64,
    pub(crate) search_stream_identity: u64,
    pub(crate) active_pool_identity: u64,
    #[cfg(not(feature = "hip-native-kernels"))]
    pub(crate) cuda_build_identity: u64,
    #[cfg(feature = "hip-native-kernels")]
    pub(crate) hip_build_identity: u64,
    pub(crate) kernel_semantics_identity: u64,
    pub(crate) binary64_math_identity: u64,
    pub(crate) plan_identity: u64,
    pub(crate) run_identity: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2ValidatedRuntimeAuthorityV2 {
    scoring_archive_layout: ResidentScoringArchiveArenaLayoutV2,
    calibration: ResidentSearchSlice2CalibrationBindingV2,
    measured_admission: ResidentSearchSlice2MeasuredAdmissionV2,
    sealed_full_workspace_receipt_identity: u64,
    sealed_post_trim_receipt_identity: u64,
    population_count: u64,
    archive_capacity: u64,
    signature_word_count: u32,
    novelty_neighbor_count: u32,
    max_terms_per_gene: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2NativeBindAuthorityV2 {
    raw: RawResidentArchiveKnnBindV2,
    measured_admission: ResidentSearchSlice2MeasuredAdmissionV2,
}

impl ResidentSearchSlice2NativeBindAuthorityV2 {
    pub(crate) fn with_raw_v2<R>(
        &self,
        use_raw: impl FnOnce(&RawResidentArchiveKnnBindV2) -> R,
    ) -> R {
        use_raw(&self.raw)
    }
}

impl ResidentSearchSlice2ValidatedRuntimeAuthorityV2 {
    /// Called only after `query_resident_search_combined_v2` and its existing
    /// receipt seal. The caller must next query/seal the FULL Slice2 allocation
    /// with this binding before create; the preliminary receipt is not enough.
    pub(crate) fn from_native_preliminary_v2(
        plan: &SealedResidentGenerationPlanV1,
        preliminary: &SealedResidentSearchAdmissionV2,
        input: &ResidentSearchSlice2InputIdentityV3,
        archive_capacity: u64,
        novelty_neighbor_count: u32,
    ) -> Result<Self, &'static str> {
        let raw = &preliminary.raw;
        let runtime = &raw.runtime;
        let binding = input.bind_plan_v3(plan)?;
        let population_count = plan.logical_population_count_v1();
        let signature_word_count =
            validate_native_shape_v2(plan.feature_count_v1(), plan.max_terms_per_gene_v1())?;
        let checked = seal_combined_search_admission_v2(*raw)
            .map_err(|_| "Slice2 preliminary native allocation receipt is inconsistent")?;
        let runtime_identity = runtime
            .identity_v3()
            .map_err(|_| "Slice2 runtime facts are invalid")?;
        #[cfg(feature = "hip-native-kernels")]
        {
            let lease = std::num::NonZeroU64::new(runtime.owner.lease_id)
                .ok_or("HIP runtime lease is zero")?;
            let actual =
                crate::hip_runtime_v1::validate_facts_v1(&runtime.owner, lease, binding.ordinal)
                    .map_err(|_| "HIP runtime facts differ from the selected owner")?
                    .0;
            if actual != binding.hip_runtime {
                return Err("HIP combined admission belongs to another physical owner");
            }
        }
        if runtime_identity.ordinal != binding.ordinal
            || binding
                .device_uuid
                .is_some_and(|uuid| uuid != runtime_identity.device_uuid)
            || raw.abi_version != crate::resident_scoring_v2::selected_combined_abi_v3()
            || raw.flags != 0
            || raw.free_memory_snapshot_count != 1
            || raw.generation.logical_population_count != population_count
            || raw.generation.allocation_plan_sha256
                != plan
                    .adaptive_policy_identity_sha256_v3()
                    .unwrap_or_else(|| plan.plan_identity_sha256_v1())
            || preliminary.generation_allocation_plan_sha256
                != plan
                    .adaptive_policy_identity_sha256_v3()
                    .unwrap_or_else(|| plan.plan_identity_sha256_v1())
            || preliminary.receipt_identity_sha256 == [0; 32]
            || preliminary.receipt_identity_sha256 != raw.receipt_identity_sha256
            || preliminary.receipt_identity_sha256 != checked.receipt_identity_sha256
            || raw.free_bytes_v3() == 0
            || raw.free_bytes_v3() > raw.total_bytes_v3()
            || raw.full_discovery_reserve_bytes != runtime.allocator_context_reserve_bytes
            || raw.pool_reserved_current_bytes != runtime.pool_reserved_bytes_v3()
            || raw.pool_used_current_bytes != runtime.pool_used_bytes_v3()
        {
            return Err("Slice2 preliminary native admission does not match the sealed run");
        }
        if plan.run_identity_sha256_v1() == [0; 32]
            || plan.generation_semantics_sha256_v1() == [0; 32]
        {
            return Err("Slice2 trim/workspace provenance does not match the sealed run");
        }
        let scoring_archive_layout = ResidentScoringArchiveArenaLayoutV2::from_native_scratch_v2(
            population_count,
            archive_capacity,
            raw.scoring.cub_scratch_bytes_v2(),
            signature_word_count,
            novelty_neighbor_count,
        )?;
        let archive_bytes = scoring_archive_layout.total_device_bytes_v2();
        validate_combined_bytes_v2(
            raw.free_bytes_v3(),
            runtime.allocator_context_reserve_bytes,
            raw.generation_device_bytes,
            archive_bytes,
        )?;
        let calibration = ResidentSearchSlice2CalibrationBindingV2 {
            device_uuid: runtime_identity.device_uuid,
            #[cfg(not(feature = "hip-native-kernels"))]
            primary_context_identity: runtime_identity.owner_identity,
            #[cfg(feature = "hip-native-kernels")]
            hip_lease_identity: runtime_identity.owner_identity,
            search_stream_identity: runtime_identity.stream_identity,
            active_pool_identity: runtime_identity.pool_identity,
            #[cfg(not(feature = "hip-native-kernels"))]
            cuda_build_identity: native_identity_v2(&plan.native_build_manifest_sha256_v1()),
            #[cfg(feature = "hip-native-kernels")]
            hip_build_identity: native_identity_v2(&plan.native_build_manifest_sha256_v1()),
            kernel_semantics_identity: native_identity_v2(&plan.generation_semantics_sha256_v1()),
            binary64_math_identity: native_identity_v2(&binding.math_sha256),
            plan_identity: native_identity_v2(&plan.plan_identity_sha256_v1()),
            run_identity: native_identity_v2(&plan.run_identity_sha256_v1()),
        };
        Ok(Self {
            scoring_archive_layout,
            calibration,
            measured_admission: ResidentSearchSlice2MeasuredAdmissionV2 {
                preliminary_receipt_sha256: preliminary.receipt_identity_sha256,
                import_admission_sha256: binding.admission_sha256,
                post_trim_plan_sha256: binding.selection_sha256,
                same_context_free_bytes: raw.free_bytes_v3(),
                allocator_context_reserve_bytes: runtime.allocator_context_reserve_bytes,
                generation_device_bytes: raw.generation_device_bytes,
                scoring_archive_device_bytes: archive_bytes,
            },
            sealed_full_workspace_receipt_identity: native_identity_v2(&binding.admission_sha256),
            sealed_post_trim_receipt_identity: native_identity_v2(&binding.selection_sha256),
            population_count,
            archive_capacity,
            signature_word_count,
            novelty_neighbor_count,
            max_terms_per_gene: plan.max_terms_per_gene_v1(),
        })
    }

    pub(crate) fn into_native_bind_authority_v2(self) -> ResidentSearchSlice2NativeBindAuthorityV2 {
        let Self {
            scoring_archive_layout,
            calibration,
            measured_admission,
            sealed_full_workspace_receipt_identity,
            sealed_post_trim_receipt_identity,
            population_count,
            archive_capacity,
            signature_word_count,
            novelty_neighbor_count,
            max_terms_per_gene,
        } = self;
        let raw = scoring_archive_layout.into_native_bind_v2(
            calibration,
            population_count,
            archive_capacity,
            signature_word_count,
            novelty_neighbor_count,
            max_terms_per_gene,
            sealed_full_workspace_receipt_identity,
            sealed_post_trim_receipt_identity,
        );
        ResidentSearchSlice2NativeBindAuthorityV2 {
            raw,
            measured_admission,
        }
    }
}

fn validate_native_shape_v2(feature_count: u64, max_terms: u32) -> Result<u32, &'static str> {
    if feature_count == 0 {
        return Err("Slice2 archive signatures require a nonempty feature vocabulary");
    }
    if !(1..=16).contains(&max_terms) || u64::from(max_terms) > feature_count {
        return Err("Slice2 active term limit must be 1..=16 and fit the feature vocabulary");
    }
    // Four words retain the existing disjoint CUB key/value workspace. Wider
    // vocabularies use every word; the archive's padded term stride stays 16.
    u32::try_from(feature_count.div_ceil(64).max(4))
        .map_err(|_| "Slice2 signature word count exceeds the native u32 domain")
}

fn validate_combined_bytes_v2(
    free: u64,
    reserve: u64,
    generation: u64,
    archive: u64,
) -> Result<(), &'static str> {
    if reserve == 0 || generation == 0 || archive == 0 {
        return Err("Slice2 measured reserve and allocation extents must be nonzero");
    }
    let required = reserve
        .checked_add(generation)
        .and_then(|n| n.checked_add(archive))
        .ok_or("Slice2 combined allocation overflow")?;
    if required > free {
        return Err(
            "Slice2 combined generation/archive allocation exceeds same-context free memory",
        );
    }
    Ok(())
}

// EXACT native generation run-token reduction, including its zero convention.
// These are compact process-local handles, not substitutes for retained SHA256.
fn native_identity_v2(identity: &[u8; 32]) -> u64 {
    identity
        .iter()
        .fold(14_695_981_039_346_656_037_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(1_099_511_628_211)
        })
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice2_native_admission_preserves_full_vocabulary_and_active_term_limit() {
        for (features, terms, words) in [
            (1, 1, 4),
            (64, 12, 4),
            (256, 16, 4),
            (257, 12, 5),
            (1_924, 12, 31),
            (u64::from(u32::MAX) * 64, 16, u32::MAX),
        ] {
            assert_eq!(validate_native_shape_v2(features, terms), Ok(words));
        }
        for features in [0, u64::from(u32::MAX) * 64 + 1, u64::MAX] {
            assert!(validate_native_shape_v2(features, 16).is_err());
        }
        for terms in [0, 17, u32::MAX] {
            assert!(validate_native_shape_v2(256, terms).is_err());
        }
        assert!(validate_native_shape_v2(11, 12).is_err());
        for terms in 1..=16 {
            assert_eq!(validate_native_shape_v2(1_924, terms), Ok(31));
        }
    }

    #[test]
    fn slice2_native_admission_charges_both_arenas_and_context_reserve() {
        assert!(validate_combined_bytes_v2(1_000, 100, 300, 600).is_ok());
        assert!(validate_combined_bytes_v2(999, 100, 300, 600).is_err());
        assert!(validate_combined_bytes_v2(u64::MAX, 1, u64::MAX, 1).is_err());
        for (reserve, generation, archive) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert!(validate_combined_bytes_v2(100, reserve, generation, archive).is_err());
        }
    }

    #[test]
    fn slice2_native_admission_identity_matches_native_run_token_algorithm() {
        let identity = std::array::from_fn(|i| i as u8);
        let mut expected = 14_695_981_039_346_656_037_u64;
        for byte in identity {
            expected ^= u64::from(byte);
            expected = expected.wrapping_mul(1_099_511_628_211);
        }
        assert_eq!(native_identity_v2(&identity), expected.max(1));
        let mut changed = identity;
        changed[31] ^= 1;
        assert_ne!(native_identity_v2(&identity), native_identity_v2(&changed));
        let source = include_str!("../native/resident_generation_v1.cu");
        assert!(source.contains("FNV_OFFSET_0_V1 = 14695981039346656037ull"));
        assert!(source.contains("FNV_PRIME_V1 = 1099511628211ull"));
        assert!(
            source
                .contains("(created->run_token ^ plan->run_identity_sha256[index]) * FNV_PRIME_V1")
        );
        assert!(source.contains("if (created->run_token == 0)"));
    }
}
