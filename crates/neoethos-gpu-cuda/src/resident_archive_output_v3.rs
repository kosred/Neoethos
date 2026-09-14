//! Terminal-only host output of committed archive and last evaluated population.
//!
//! The two sources remain distinct so Search can apply its existing exact
//! archive/population union, retaining the final evaluation for duplicates.
//! The native owner supplies terminal event proof; this module admits buffers
//! and validates the complete copy before making any candidate accessible.

use std::collections::HashSet;

use neoethos_gpu_contracts::device::NeoPopulationMetricRow;
use thiserror::Error;

pub(crate) const RESIDENT_ARCHIVE_EXPORT_ABI_V4: u32 = 4;
pub(crate) const RESIDENT_POPULATION_EXPORT_ABI_V3: u32 = 3;
pub(crate) const RESIDENT_ARCHIVE_TERM_STRIDE_V3: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RawResidentPopulationExportReceiptV3 {
    pub(crate) abi_version: u32,
    pub(crate) reserved: u32,
    pub(crate) run_identity: u64,
    pub(crate) packed_commit_word: u64,
    pub(crate) evaluated_generation: u64,
    pub(crate) candidate_count: u64,
    pub(crate) term_count: u64,
    pub(crate) feature_count: u64,
    pub(crate) host_copy_count: u64,
    pub(crate) host_copy_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResidentPopulationExportContextV3 {
    pub(crate) run_identity: u64,
    pub(crate) packed_commit_word: u64,
    pub(crate) candidate_count: u64,
    pub(crate) feature_count: u64,
    pub(crate) terminal_generation: u64,
    pub(crate) evaluated_generation: u64,
    pub(crate) max_terms: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RawResidentArchiveGeneScalarV3 {
    pub(crate) gene_identity: u64,
    pub(crate) content_hash: u64,
    pub(crate) term_count: u32,
    pub(crate) smc_flags: u32,
    pub(crate) long_threshold: f64,
    pub(crate) short_threshold: f64,
    pub(crate) target_pips: f64,
    pub(crate) stop_pips: f64,
    pub(crate) stop_vol_multiplier: f64,
    pub(crate) generation: u32,
    pub(crate) reserved: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RawResidentArchiveExportReceiptV3 {
    pub(crate) abi_version: u32,
    pub(crate) reserved: u32,
    pub(crate) run_identity: u64,
    pub(crate) packed_commit_word: u64,
    pub(crate) candidate_count: u64,
    pub(crate) term_count: u64,
    pub(crate) feature_count: u64,
    pub(crate) host_copy_count: u64,
    pub(crate) host_copy_bytes: u64,
}

const _: [(); 72] = [(); std::mem::size_of::<RawResidentArchiveGeneScalarV3>()];
const _: [(); 8] = [(); std::mem::align_of::<RawResidentArchiveGeneScalarV3>()];
const _: [(); 64] = [(); std::mem::size_of::<RawResidentArchiveExportReceiptV3>()];
const _: [(); 8] = [(); std::mem::align_of::<RawResidentArchiveExportReceiptV3>()];
const _: [(); 104] = [(); std::mem::size_of::<NeoPopulationMetricRow>()];
const _: [(); 72] = [(); std::mem::size_of::<RawResidentPopulationExportReceiptV3>()];
const _: [(); 8] = [(); std::mem::align_of::<RawResidentPopulationExportReceiptV3>()];

/// Values copied from the native owner's already verified terminal authority.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResidentArchiveExportContextV3 {
    pub(crate) run_identity: u64,
    pub(crate) packed_commit_word: u64,
    pub(crate) candidate_count: u64,
    pub(crate) feature_count: u64,
    pub(crate) max_terms: u32,
    pub(crate) terminal_generation: u64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResidentArchiveOutputErrorV3 {
    #[error("resident archive output arithmetic overflow: {0}")]
    Overflow(&'static str),
    #[error("resident archive output allocation refused: {0}")]
    Allocation(&'static str),
    #[error("resident archive output authority mismatch: {0}")]
    Authority(&'static str),
    #[error("resident archive output candidate {index} is invalid: {reason}")]
    Candidate { index: usize, reason: &'static str },
}

#[derive(Debug)]
pub(crate) struct ResidentArchiveOutputBuffersV3 {
    context: ResidentArchiveExportContextV3,
    scalars: Vec<RawResidentArchiveGeneScalarV3>,
    indices: Vec<u64>,
    weights: Vec<f64>,
    metrics: Vec<NeoPopulationMetricRow>,
    admission_sequences: Vec<u64>,
    copy_bytes: u64,
    term_stride: usize,
}

pub(crate) struct ResidentArchiveOutputFfiBuffersV3 {
    pub(crate) scalars: *mut RawResidentArchiveGeneScalarV3,
    pub(crate) indices: *mut u64,
    pub(crate) weights: *mut f64,
    pub(crate) metrics: *mut NeoPopulationMetricRow,
    pub(crate) admission_sequences: *mut u64,
    pub(crate) candidate_capacity: u64,
    pub(crate) term_capacity: u64,
}

fn zeroed_vector_v3<T: Default + Clone>(
    count: usize,
    name: &'static str,
) -> Result<Vec<T>, ResidentArchiveOutputErrorV3> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| ResidentArchiveOutputErrorV3::Allocation(name))?;
    values.resize(count, T::default());
    Ok(values)
}

impl ResidentArchiveOutputBuffersV3 {
    pub(crate) fn allocate(
        context: ResidentArchiveExportContextV3,
        archive_capacity: u64,
    ) -> Result<Self, ResidentArchiveOutputErrorV3> {
        Self::allocate_with_stride(
            context,
            archive_capacity,
            RESIDENT_ARCHIVE_TERM_STRIDE_V3,
            true,
        )
    }

    fn allocate_with_stride(
        context: ResidentArchiveExportContextV3,
        capacity: u64,
        term_stride: usize,
        include_admission_sequences: bool,
    ) -> Result<Self, ResidentArchiveOutputErrorV3> {
        use ResidentArchiveOutputErrorV3 as Error;
        if context.run_identity == 0
            || context.feature_count == 0
            || context.terminal_generation == 0
        {
            return Err(Error::Authority(
                "empty run, feature extent or terminal generation",
            ));
        }
        if context.max_terms == 0 || context.max_terms as usize > RESIDENT_ARCHIVE_TERM_STRIDE_V3 {
            return Err(Error::Authority(
                "active term capacity is outside archive storage",
            ));
        }
        if term_stride < context.max_terms as usize || term_stride > RESIDENT_ARCHIVE_TERM_STRIDE_V3
        {
            return Err(Error::Authority(
                "term stride does not cover admitted active terms",
            ));
        }
        if context.candidate_count > capacity {
            return Err(Error::Authority(
                "committed count exceeds admitted archive capacity",
            ));
        }
        let count = usize::try_from(context.candidate_count)
            .map_err(|_| Error::Overflow("candidate count"))?;
        let terms = count
            .checked_mul(term_stride)
            .ok_or(Error::Overflow("term extent"))?;
        let copy_bytes = context
            .candidate_count
            .checked_mul(
                72 + 104
                    + 16 * term_stride as u64
                    + if include_admission_sequences { 8 } else { 0 },
            )
            .ok_or(Error::Overflow("copy bytes"))?;
        // All extents are checked before the first potentially large allocation.
        for (elements, width) in [(count, 72usize), (terms, 8), (count, 104)] {
            let bytes = elements
                .checked_mul(width)
                .ok_or(Error::Overflow("host extent"))?;
            if bytes > isize::MAX as usize {
                return Err(Error::Overflow("host addressable extent"));
            }
        }
        Ok(Self {
            context,
            scalars: zeroed_vector_v3(count, "scalars")?,
            indices: zeroed_vector_v3(terms, "indices")?,
            weights: zeroed_vector_v3(terms, "weights")?,
            metrics: zeroed_vector_v3(count, "metrics")?,
            admission_sequences: zeroed_vector_v3(
                if include_admission_sequences {
                    count
                } else {
                    0
                },
                "admission sequences",
            )?,
            copy_bytes,
            term_stride,
        })
    }

    /// Pointers remain valid only while this owner is retained and not moved
    /// into `seal`. Native must honor both capacities and complete its copies.
    pub(crate) fn ffi_buffers_mut(&mut self) -> ResidentArchiveOutputFfiBuffersV3 {
        ResidentArchiveOutputFfiBuffersV3 {
            scalars: self.scalars.as_mut_ptr(),
            indices: self.indices.as_mut_ptr(),
            weights: self.weights.as_mut_ptr(),
            metrics: self.metrics.as_mut_ptr(),
            admission_sequences: if self.admission_sequences.is_empty() {
                std::ptr::null_mut()
            } else {
                self.admission_sequences.as_mut_ptr()
            },
            candidate_capacity: self.scalars.len() as u64,
            term_capacity: self.indices.len() as u64,
        }
    }

    /// Validate the complete copied archive against the retained terminal and
    /// scenario authority. This does not manufacture device execution proof or
    /// recompute the native content hash; both remain bound by the native owner.
    pub(crate) fn seal(
        self,
        receipt: RawResidentArchiveExportReceiptV3,
        allowed_scenario_ids: &[u64],
    ) -> Result<ResidentArchiveTerminalOutputV3, ResidentArchiveOutputErrorV3> {
        use ResidentArchiveOutputErrorV3 as Error;
        if receipt.abi_version != RESIDENT_ARCHIVE_EXPORT_ABI_V4
            || receipt.reserved != 0
            || receipt.run_identity != self.context.run_identity
            || receipt.packed_commit_word != self.context.packed_commit_word
            || receipt.candidate_count != self.context.candidate_count
            || receipt.term_count != self.indices.len() as u64
            || receipt.feature_count != self.context.feature_count
            || receipt.host_copy_bytes != self.copy_bytes
            || receipt.host_copy_count != if self.scalars.is_empty() { 0 } else { 5 }
            || self.admission_sequences.len() != self.scalars.len()
        {
            return Err(Error::Authority("terminal export receipt fields"));
        }
        self.validate_candidates(allowed_scenario_ids, false)?;
        // Slot reuse must not become a new tie-break priority. Retain native
        // admission order through an index permutation, never by reconstructing
        // or partially moving a candidate's gene and metric fields.
        let mut admission_order = zeroed_vector_v3(self.scalars.len(), "admission order")?;
        for (index, slot) in admission_order.iter_mut().enumerate() {
            *slot = index;
        }
        admission_order.sort_unstable_by_key(|&index| self.admission_sequences[index]);
        if admission_order
            .windows(2)
            .any(|pair| self.admission_sequences[pair[0]] == self.admission_sequences[pair[1]])
        {
            return Err(Error::Authority("duplicate archive admission sequence"));
        }
        Ok(ResidentArchiveTerminalOutputV3 {
            buffers: self,
            receipt,
            admission_order,
        })
    }

    fn validate_candidates(
        &self,
        allowed_scenario_ids: &[u64],
        allow_economic_rejection: bool,
    ) -> Result<(), ResidentArchiveOutputErrorV3> {
        use ResidentArchiveOutputErrorV3 as Error;
        let mut scenarios = HashSet::new();
        scenarios
            .try_reserve(allowed_scenario_ids.len())
            .map_err(|_| Error::Allocation("scenario identities"))?;
        scenarios.extend(allowed_scenario_ids.iter().copied());
        let mut identities = HashSet::new();
        identities
            .try_reserve(self.scalars.len())
            .map_err(|_| Error::Allocation("candidate identities"))?;
        for (index, (scalar, metric)) in self.scalars.iter().zip(&self.metrics).enumerate() {
            let reject = |reason| Error::Candidate { index, reason };
            let count = scalar.term_count as usize;
            if scalar.reserved != 0
                || count == 0
                || count > RESIDENT_ARCHIVE_TERM_STRIDE_V3
                || scalar.term_count > self.context.max_terms
                || scalar.smc_flags & !0x7ff != 0
            {
                return Err(reject("scalar shape or unknown flags"));
            }
            if u64::from(scalar.generation) >= self.context.terminal_generation {
                return Err(reject(
                    "archive member was not evaluated before terminal publication",
                ));
            }
            if !identities.insert(scalar.gene_identity)
                || metric.candidate_id != scalar.gene_identity
                || !scenarios.contains(&metric.scenario_id)
            {
                return Err(reject("candidate or scenario identity"));
            }
            if ![
                scalar.long_threshold,
                scalar.short_threshold,
                scalar.target_pips,
                scalar.stop_pips,
                scalar.stop_vol_multiplier,
            ]
            .into_iter()
            .all(f64::is_finite)
                || scalar.long_threshold <= scalar.short_threshold
                || scalar.target_pips <= 0.0
                || scalar.stop_pips <= 0.0
                || scalar.stop_vol_multiplier < 0.0
            {
                return Err(reject("non-finite or invalid signal/exit geometry"));
            }
            if allow_economic_rejection {
                // Only the population's checked terminal producer may supply
                // the monthly-equity rejection marker. It is not permission
                // to hide NaN, +infinity or faults in any other metric slot.
                if neoethos_gpu_contracts::resident_search_scoring_v2::classify_resident_metrics_v2(
                    &metric.values,
                )
                .is_err()
                {
                    return Err(reject("invalid evaluated metric row"));
                }
            } else if !metric.values.iter().all(|value| value.is_finite()) {
                return Err(reject("non-finite metric row"));
            }
            let base = index * self.term_stride;
            for term in 0..self.term_stride {
                let feature = self.indices[base + term];
                let weight = self.weights[base + term];
                if term < count {
                    if feature >= self.context.feature_count || !weight.is_finite() {
                        return Err(reject("feature extent or non-finite weight"));
                    }
                } else if feature != 0 || weight.to_bits() != 0 {
                    return Err(reject("noncanonical term padding"));
                }
            }
        }
        Ok(())
    }

    fn candidate(&self, index: usize) -> Option<ResidentArchiveCandidateV3<'_>> {
        let scalar = self.scalars.get(index)?;
        let start = index * self.term_stride;
        let end = start + scalar.term_count as usize;
        Some(ResidentArchiveCandidateV3 {
            scalar,
            indices: &self.indices[start..end],
            weights: &self.weights[start..end],
            metric: &self.metrics[index],
        })
    }
}

/// Every committed archive member in original admission order, not a top-K list.
/// Does not include the last evaluated population and is not a validation pass.
#[derive(Debug)]
pub struct ResidentArchiveTerminalOutputV3 {
    buffers: ResidentArchiveOutputBuffersV3,
    receipt: RawResidentArchiveExportReceiptV3,
    admission_order: Vec<usize>,
}

impl ResidentArchiveTerminalOutputV3 {
    pub fn len(&self) -> usize {
        self.buffers.scalars.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buffers.scalars.is_empty()
    }
    pub fn host_copy_count(&self) -> u64 {
        self.receipt.host_copy_count
    }
    pub fn host_copy_bytes(&self) -> u64 {
        self.receipt.host_copy_bytes
    }
    pub fn run_identity(&self) -> u64 {
        self.receipt.run_identity
    }
    pub fn packed_commit_word(&self) -> u64 {
        self.receipt.packed_commit_word
    }
    pub fn feature_count(&self) -> u64 {
        self.receipt.feature_count
    }
    pub fn terminal_generation(&self) -> u64 {
        self.buffers.context.terminal_generation
    }

    pub fn candidate(&self, index: usize) -> Option<ResidentArchiveCandidateV3<'_>> {
        self.buffers.candidate(*self.admission_order.get(index)?)
    }

    pub fn candidates(&self) -> impl ExactSizeIterator<Item = ResidentArchiveCandidateV3<'_>> {
        (0..self.len()).map(|index| self.candidate(index).expect("sealed archive index"))
    }
}

/// Retains all P evaluated rows at the generation's actual K-slot stride.
/// Kept separate from the archive so its unchanged profitability admission
/// cannot suppress the remaining strategies before downstream validation.
#[derive(Debug)]
pub(crate) struct ResidentPopulationOutputBuffersV3 {
    context: ResidentPopulationExportContextV3,
    buffers: ResidentArchiveOutputBuffersV3,
}

impl ResidentPopulationOutputBuffersV3 {
    pub(crate) fn allocate(
        context: ResidentPopulationExportContextV3,
    ) -> Result<Self, ResidentArchiveOutputErrorV3> {
        if context.candidate_count == 0
            || context.terminal_generation.checked_sub(1) != Some(context.evaluated_generation)
        {
            return Err(ResidentArchiveOutputErrorV3::Authority(
                "population must be the nonempty last evaluated generation",
            ));
        }
        let buffers = ResidentArchiveOutputBuffersV3::allocate_with_stride(
            ResidentArchiveExportContextV3 {
                run_identity: context.run_identity,
                packed_commit_word: context.packed_commit_word,
                candidate_count: context.candidate_count,
                feature_count: context.feature_count,
                max_terms: context.max_terms,
                terminal_generation: context.terminal_generation,
            },
            context.candidate_count,
            context.max_terms as usize,
            false,
        )?;
        Ok(Self { context, buffers })
    }

    pub(crate) fn ffi_buffers_mut(&mut self) -> ResidentArchiveOutputFfiBuffersV3 {
        self.buffers.ffi_buffers_mut()
    }

    pub(crate) fn seal(
        self,
        receipt: RawResidentPopulationExportReceiptV3,
        allowed_scenario_ids: &[u64],
    ) -> Result<ResidentPopulationTerminalOutputV3, ResidentArchiveOutputErrorV3> {
        if receipt.abi_version != RESIDENT_POPULATION_EXPORT_ABI_V3
            || receipt.reserved != 0
            || receipt.run_identity != self.context.run_identity
            || receipt.packed_commit_word != self.context.packed_commit_word
            || receipt.evaluated_generation != self.context.evaluated_generation
            || receipt.candidate_count != self.context.candidate_count
            || receipt.term_count != self.buffers.indices.len() as u64
            || receipt.feature_count != self.context.feature_count
            || receipt.host_copy_count != 4
            || receipt.host_copy_bytes != self.buffers.copy_bytes
        {
            return Err(ResidentArchiveOutputErrorV3::Authority(
                "evaluated population export receipt fields",
            ));
        }
        self.buffers
            .validate_candidates(allowed_scenario_ids, true)?;
        Ok(ResidentPopulationTerminalOutputV3 {
            buffers: self,
            receipt,
        })
    }
}

/// Complete last evaluated population, not the newly generated offspring.
/// Sealing proves transport/identity integrity; it is not a profitability,
/// walk-forward, out-of-sample or promotion decision.
#[derive(Debug)]
pub struct ResidentPopulationTerminalOutputV3 {
    buffers: ResidentPopulationOutputBuffersV3,
    receipt: RawResidentPopulationExportReceiptV3,
}

impl ResidentPopulationTerminalOutputV3 {
    pub fn len(&self) -> usize {
        self.buffers.buffers.scalars.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buffers.buffers.scalars.is_empty()
    }
    pub fn run_identity(&self) -> u64 {
        self.receipt.run_identity
    }
    pub fn packed_commit_word(&self) -> u64 {
        self.receipt.packed_commit_word
    }
    pub fn evaluated_generation(&self) -> u64 {
        self.receipt.evaluated_generation
    }
    pub fn feature_count(&self) -> u64 {
        self.receipt.feature_count
    }
    pub fn max_terms(&self) -> u32 {
        self.buffers.context.max_terms
    }
    pub fn host_copy_count(&self) -> u64 {
        self.receipt.host_copy_count
    }
    pub fn host_copy_bytes(&self) -> u64 {
        self.receipt.host_copy_bytes
    }
    pub fn candidate(&self, index: usize) -> Option<ResidentArchiveCandidateV3<'_>> {
        self.buffers.buffers.candidate(index)
    }
    pub fn candidates(&self) -> impl ExactSizeIterator<Item = ResidentArchiveCandidateV3<'_>> {
        (0..self.len()).map(|index| self.candidate(index).expect("sealed population index"))
    }
}

/// One terminal collection retaining both bounded sources without re-scoring
/// or truncating. Search owns its canonical exact-behavior deduplication.
#[derive(Debug)]
pub struct ResidentSearchTerminalCandidatesV3 {
    archive: ResidentArchiveTerminalOutputV3,
    population: ResidentPopulationTerminalOutputV3,
}

impl ResidentSearchTerminalCandidatesV3 {
    pub(crate) fn seal(
        archive: ResidentArchiveTerminalOutputV3,
        population: ResidentPopulationTerminalOutputV3,
    ) -> Result<Self, ResidentArchiveOutputErrorV3> {
        if archive.run_identity() != population.run_identity()
            || archive.packed_commit_word() != population.packed_commit_word()
            || archive.feature_count() != population.feature_count()
            || archive.terminal_generation().checked_sub(1)
                != Some(population.evaluated_generation())
            || archive.buffers.context.max_terms != population.max_terms()
        {
            return Err(ResidentArchiveOutputErrorV3::Authority(
                "archive and population belong to different terminal runs",
            ));
        }
        Ok(Self {
            archive,
            population,
        })
    }
    pub fn archive(&self) -> &ResidentArchiveTerminalOutputV3 {
        &self.archive
    }
    pub fn population(&self) -> &ResidentPopulationTerminalOutputV3 {
        &self.population
    }
    pub fn into_parts(
        self,
    ) -> (
        ResidentArchiveTerminalOutputV3,
        ResidentPopulationTerminalOutputV3,
    ) {
        (self.archive, self.population)
    }
}

#[derive(Debug)]
pub struct ResidentArchiveCandidateV3<'a> {
    scalar: &'a RawResidentArchiveGeneScalarV3,
    indices: &'a [u64],
    weights: &'a [f64],
    metric: &'a NeoPopulationMetricRow,
}

impl ResidentArchiveCandidateV3<'_> {
    pub fn gene_identity(&self) -> u64 {
        self.scalar.gene_identity
    }
    pub fn content_hash(&self) -> u64 {
        self.scalar.content_hash
    }
    pub fn generation(&self) -> u32 {
        self.scalar.generation
    }
    pub fn smc_flags(&self) -> u32 {
        self.scalar.smc_flags
    }
    pub fn long_threshold(&self) -> f64 {
        self.scalar.long_threshold
    }
    pub fn short_threshold(&self) -> f64 {
        self.scalar.short_threshold
    }
    pub fn target_pips(&self) -> f64 {
        self.scalar.target_pips
    }
    pub fn stop_pips(&self) -> f64 {
        self.scalar.stop_pips
    }
    pub fn stop_vol_multiplier(&self) -> f64 {
        self.scalar.stop_vol_multiplier
    }
    pub fn indices(&self) -> &[u64] {
        self.indices
    }
    pub fn weights(&self) -> &[f64] {
        self.weights
    }
    pub fn metric_row(&self) -> &NeoPopulationMetricRow {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(
        count: usize,
    ) -> (
        ResidentArchiveOutputBuffersV3,
        RawResidentArchiveExportReceiptV3,
    ) {
        let context = ResidentArchiveExportContextV3 {
            run_identity: 77,
            packed_commit_word: 1234,
            candidate_count: count as u64,
            feature_count: 1024,
            max_terms: 12,
            terminal_generation: 5,
        };
        let mut buffers = ResidentArchiveOutputBuffersV3::allocate(context, count as u64).unwrap();
        for i in 0..count {
            buffers.admission_sequences[i] = i as u64;
            buffers.scalars[i] = RawResidentArchiveGeneScalarV3 {
                gene_identity: i as u64,
                content_hash: 900 + i as u64,
                term_count: 2,
                smc_flags: 0x401,
                long_threshold: 0.75,
                short_threshold: -0.5,
                target_pips: 0.75,
                stop_pips: 0.25,
                stop_vol_multiplier: 1.5,
                generation: (i % 5) as u32,
                reserved: 0,
            };
            buffers.indices[i * 16] = 900;
            buffers.indices[i * 16 + 1] = 3;
            buffers.weights[i * 16] = -0.0;
            buffers.weights[i * 16 + 1] = -0.125;
            buffers.metrics[i] = NeoPopulationMetricRow {
                candidate_id: i as u64,
                scenario_id: 314,
                values: [
                    125.5, 1.2, 10_125.5, 0.1, 0.75, 2.0, 12.55, 0.5, 10.0, 0.6, 0.2,
                ],
            };
        }
        let receipt = RawResidentArchiveExportReceiptV3 {
            abi_version: 4,
            reserved: 0,
            run_identity: 77,
            packed_commit_word: 1234,
            candidate_count: count as u64,
            term_count: count as u64 * 16,
            feature_count: 1024,
            host_copy_count: if count == 0 { 0 } else { 5 },
            host_copy_bytes: count as u64 * 440,
        };
        (buffers, receipt)
    }

    #[test]
    fn terminal_archive_output_abi_matches_native_layout() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<RawResidentArchiveGeneScalarV3>(), 72);
        assert_eq!(align_of::<RawResidentArchiveGeneScalarV3>(), 8);
        assert_eq!(offset_of!(RawResidentArchiveGeneScalarV3, gene_identity), 0);
        assert_eq!(offset_of!(RawResidentArchiveGeneScalarV3, content_hash), 8);
        assert_eq!(offset_of!(RawResidentArchiveGeneScalarV3, term_count), 16);
        assert_eq!(offset_of!(RawResidentArchiveGeneScalarV3, smc_flags), 20);
        assert_eq!(
            offset_of!(RawResidentArchiveGeneScalarV3, long_threshold),
            24
        );
        assert_eq!(
            offset_of!(RawResidentArchiveGeneScalarV3, short_threshold),
            32
        );
        assert_eq!(offset_of!(RawResidentArchiveGeneScalarV3, target_pips), 40);
        assert_eq!(offset_of!(RawResidentArchiveGeneScalarV3, stop_pips), 48);
        assert_eq!(
            offset_of!(RawResidentArchiveGeneScalarV3, stop_vol_multiplier),
            56
        );
        assert_eq!(offset_of!(RawResidentArchiveGeneScalarV3, generation), 64);
        assert_eq!(offset_of!(RawResidentArchiveGeneScalarV3, reserved), 68);
        assert_eq!(size_of::<RawResidentArchiveExportReceiptV3>(), 64);
        assert_eq!(align_of::<RawResidentArchiveExportReceiptV3>(), 8);
        assert_eq!(
            offset_of!(RawResidentArchiveExportReceiptV3, abi_version),
            0
        );
        assert_eq!(offset_of!(RawResidentArchiveExportReceiptV3, reserved), 4);
        assert_eq!(
            offset_of!(RawResidentArchiveExportReceiptV3, run_identity),
            8
        );
        assert_eq!(
            offset_of!(RawResidentArchiveExportReceiptV3, packed_commit_word),
            16
        );
        assert_eq!(
            offset_of!(RawResidentArchiveExportReceiptV3, candidate_count),
            24
        );
        assert_eq!(
            offset_of!(RawResidentArchiveExportReceiptV3, term_count),
            32
        );
        assert_eq!(
            offset_of!(RawResidentArchiveExportReceiptV3, feature_count),
            40
        );
        assert_eq!(
            offset_of!(RawResidentArchiveExportReceiptV3, host_copy_count),
            48
        );
        assert_eq!(
            offset_of!(RawResidentArchiveExportReceiptV3, host_copy_bytes),
            56
        );
    }

    #[test]
    fn terminal_archive_output_preserves_every_member_and_exact_values() {
        let (mut buffers, receipt) = fixture(73);
        let ffi = buffers.ffi_buffers_mut();
        assert_eq!((ffi.candidate_capacity, ffi.term_capacity), (73, 1168));
        assert_eq!(ffi.scalars, buffers.scalars.as_mut_ptr());
        assert_eq!(ffi.indices, buffers.indices.as_mut_ptr());
        assert_eq!(ffi.weights, buffers.weights.as_mut_ptr());
        assert_eq!(ffi.metrics, buffers.metrics.as_mut_ptr());
        assert_eq!(
            ffi.admission_sequences,
            buffers.admission_sequences.as_mut_ptr()
        );
        let output = buffers.seal(receipt, &[314, 2718]).unwrap();
        assert_eq!(output.len(), 73);
        assert_eq!(output.host_copy_count(), 5);
        assert_eq!(output.host_copy_bytes(), 73 * 440);
        for (i, candidate) in output.candidates().enumerate() {
            assert_eq!(candidate.gene_identity(), i as u64);
            assert_eq!(candidate.content_hash(), 900 + i as u64);
            assert_eq!(candidate.generation(), (i % 5) as u32);
            assert_eq!(candidate.indices(), &[900, 3]);
            assert_eq!(candidate.weights()[0].to_bits(), (-0.0f64).to_bits());
            assert_eq!(candidate.weights()[1].to_bits(), (-0.125f64).to_bits());
            assert_eq!(candidate.stop_pips().to_bits(), 0.25f64.to_bits());
            assert_eq!(candidate.target_pips().to_bits(), 0.75f64.to_bits());
            assert_eq!(candidate.long_threshold().to_bits(), 0.75f64.to_bits());
            assert_eq!(candidate.short_threshold().to_bits(), (-0.5f64).to_bits());
            assert_eq!(candidate.stop_vol_multiplier().to_bits(), 1.5f64.to_bits());
            assert_eq!(candidate.smc_flags(), 0x401);
            assert_eq!(candidate.metric_row().candidate_id, i as u64);
            assert_eq!(candidate.metric_row().scenario_id, 314);
            assert_eq!(
                candidate.metric_row().values.map(f64::to_bits),
                [
                    125.5f64, 1.2, 10_125.5, 0.1, 0.75, 2.0, 12.55, 0.5, 10.0, 0.6, 0.2
                ]
                .map(f64::to_bits)
            );
        }
        assert!(output.candidate(73).is_none());
    }

    #[test]
    fn terminal_archive_output_empty_has_no_copies_and_no_candidates() {
        let (buffers, receipt) = fixture(0);
        let output = buffers.seal(receipt, &[]).unwrap();
        assert!(output.is_empty());
        assert_eq!(output.host_copy_count(), 0);
        assert_eq!(output.host_copy_bytes(), 0);
    }

    #[test]
    fn terminal_archive_reused_slots_preserve_admission_priority_and_whole_observations() {
        let (mut buffers, receipt) = fixture(4);
        buffers
            .admission_sequences
            .copy_from_slice(&[101, 2, 900, 0]);
        for index in 0..4 {
            buffers.scalars[index].target_pips = index as f64 + 1.5;
            buffers.indices[index * 16] = 700 + index as u64;
            buffers.weights[index * 16] = -(index as f64) / 8.0;
            buffers.metrics[index].values[0] = 100.0 + index as f64;
        }
        let output = buffers.seal(receipt, &[314]).unwrap();
        for (candidate, original) in output.candidates().zip([3, 1, 0, 2]) {
            assert_eq!(candidate.gene_identity(), original as u64);
            assert_eq!(candidate.content_hash(), 900 + original as u64);
            assert_eq!(candidate.target_pips(), original as f64 + 1.5);
            assert_eq!(candidate.indices()[0], 700 + original as u64);
            assert_eq!(
                candidate.weights()[0].to_bits(),
                (-(original as f64) / 8.0).to_bits()
            );
            assert_eq!(candidate.metric_row().candidate_id, original as u64);
            assert_eq!(candidate.metric_row().values[0], 100.0 + original as f64);
        }
        let (mut buffers, receipt) = fixture(4);
        buffers.admission_sequences[3] = 0;
        assert!(matches!(
            buffers.seal(receipt, &[314]),
            Err(ResidentArchiveOutputErrorV3::Authority(
                "duplicate archive admission sequence"
            ))
        ));
        let (buffers, mut receipt) = fixture(1);
        receipt.abi_version = 3;
        assert!(
            buffers.seal(receipt, &[314]).is_err(),
            "old four-copy exports cannot supply sequence authority"
        );
    }

    #[test]
    fn terminal_archive_output_preserves_active_capacity_and_full_feature_extent() {
        for max_terms in [1, 12, 16] {
            let (mut buffers, mut receipt) = fixture(1);
            buffers.context.max_terms = max_terms;
            buffers.context.feature_count = 1924;
            receipt.feature_count = 1924;
            buffers.scalars[0].term_count = max_terms;
            buffers.indices.fill(0);
            buffers.weights.fill(0.0);
            for term in 0..max_terms as usize {
                buffers.indices[term] = 1923 - term as u64;
                buffers.weights[term] = (term + 1) as f64 / 16.0;
            }
            let output = buffers.seal(receipt, &[314]).unwrap();
            let candidate = output.candidate(0).unwrap();
            assert_eq!(candidate.indices().len(), max_terms as usize);
            assert_eq!(candidate.indices()[0], 1923);
            assert_eq!(candidate.weights().len(), max_terms as usize);
            assert_eq!(output.feature_count(), 1924);
            // Active K does not change the canonical sixteen-slot archive wire.
            assert_eq!(output.host_copy_bytes(), 440);
        }
    }

    #[test]
    fn terminal_archive_output_rejects_more_terms_than_the_run_admitted() {
        let (mut buffers, receipt) = fixture(1);
        buffers.scalars[0].term_count = 13;
        assert!(matches!(
            buffers.seal(receipt, &[314]),
            Err(ResidentArchiveOutputErrorV3::Candidate { index: 0, .. })
        ));
        for max_terms in [0, 17, u32::MAX] {
            let (buffers, _) = fixture(0);
            assert!(matches!(
                ResidentArchiveOutputBuffersV3::allocate(
                    ResidentArchiveExportContextV3 {
                        max_terms,
                        ..buffers.context
                    },
                    0,
                ),
                Err(ResidentArchiveOutputErrorV3::Authority(_))
            ));
        }
    }

    #[test]
    fn terminal_archive_output_rejects_all_receipt_rebinding() {
        for field in 0..9 {
            let (buffers, mut receipt) = fixture(2);
            match field {
                0 => receipt.abi_version += 1,
                1 => receipt.reserved = 1,
                2 => receipt.run_identity += 1,
                3 => receipt.packed_commit_word += 1,
                4 => receipt.candidate_count -= 1,
                5 => receipt.term_count -= 1,
                6 => receipt.feature_count += 1,
                7 => receipt.host_copy_count -= 1,
                _ => receipt.host_copy_bytes -= 1,
            }
            assert!(matches!(
                buffers.seal(receipt, &[314]),
                Err(ResidentArchiveOutputErrorV3::Authority(_))
            ));
        }
    }

    #[test]
    fn terminal_archive_output_rejects_invalid_last_member_without_partial_output() {
        for fault in 0..13 {
            let (mut buffers, receipt) = fixture(5);
            match fault {
                0 => buffers.scalars[4].reserved = 1,
                1 => buffers.scalars[4].term_count = 17,
                2 => buffers.scalars[4].smc_flags = 1 << 11,
                3 => buffers.scalars[4].generation = 5,
                4 => buffers.scalars[4].gene_identity = 0,
                5 => buffers.metrics[4].candidate_id = 0,
                6 => buffers.metrics[4].scenario_id = 0,
                7 => buffers.scalars[4].long_threshold = f64::NAN,
                8 => buffers.scalars[4].stop_pips = 0.0,
                9 => buffers.metrics[4].values[1] = f64::NAN,
                10 => buffers.indices[4 * 16] = 1024,
                11 => buffers.weights[4 * 16] = f64::INFINITY,
                _ => buffers.weights[4 * 16 + 2] = -0.0,
            }
            assert!(matches!(
                buffers.seal(receipt, &[314]),
                Err(ResidentArchiveOutputErrorV3::Candidate { index: 4, .. })
            ));
        }
    }

    #[test]
    fn terminal_archive_output_refuses_capacity_and_overflow_before_allocation() {
        let context = ResidentArchiveExportContextV3 {
            run_identity: 1,
            packed_commit_word: 1,
            candidate_count: 5,
            feature_count: 1,
            max_terms: 1,
            terminal_generation: 1,
        };
        assert!(matches!(
            ResidentArchiveOutputBuffersV3::allocate(context, 4),
            Err(ResidentArchiveOutputErrorV3::Authority(_))
        ));
        assert!(matches!(
            ResidentArchiveOutputBuffersV3::allocate(
                ResidentArchiveExportContextV3 {
                    candidate_count: u64::MAX,
                    ..context
                },
                u64::MAX
            ),
            Err(ResidentArchiveOutputErrorV3::Overflow(_))
        ));
    }

    fn population_fixture(
        count: usize,
        max_terms: u32,
    ) -> (
        ResidentPopulationOutputBuffersV3,
        RawResidentPopulationExportReceiptV3,
    ) {
        let (archive, _) = fixture(count);
        let mut output =
            ResidentPopulationOutputBuffersV3::allocate(ResidentPopulationExportContextV3 {
                run_identity: 77,
                packed_commit_word: 1234,
                candidate_count: count as u64,
                feature_count: 1024,
                terminal_generation: 5,
                evaluated_generation: 4,
                max_terms,
            })
            .unwrap();
        output.buffers.scalars = archive.scalars;
        output.buffers.metrics = archive.metrics;
        for index in 0..count {
            output.buffers.scalars[index].term_count = 2.min(max_terms);
            for term in 0..max_terms as usize {
                output.buffers.indices[index * max_terms as usize + term] =
                    archive.indices[index * 16 + term];
                output.buffers.weights[index * max_terms as usize + term] =
                    archive.weights[index * 16 + term];
            }
        }
        let receipt = RawResidentPopulationExportReceiptV3 {
            abi_version: 3,
            reserved: 0,
            run_identity: 77,
            packed_commit_word: 1234,
            evaluated_generation: 4,
            candidate_count: count as u64,
            term_count: count as u64 * max_terms as u64,
            feature_count: 1024,
            host_copy_count: 4,
            host_copy_bytes: count as u64 * (176 + 16 * max_terms as u64),
        };
        (output, receipt)
    }

    #[test]
    fn terminal_population_receipt_has_exact_additive_wire_layout() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<RawResidentPopulationExportReceiptV3>(), 72);
        assert_eq!(align_of::<RawResidentPopulationExportReceiptV3>(), 8);
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, abi_version),
            0
        );
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, reserved),
            4
        );
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, run_identity),
            8
        );
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, packed_commit_word),
            16
        );
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, evaluated_generation),
            24
        );
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, candidate_count),
            32
        );
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, term_count),
            40
        );
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, feature_count),
            48
        );
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, host_copy_count),
            56
        );
        assert_eq!(
            offset_of!(RawResidentPopulationExportReceiptV3, host_copy_bytes),
            64
        );
    }

    #[test]
    fn terminal_population_retains_every_evaluated_row_at_actual_term_stride() {
        for max_terms in [1, 12, 16] {
            let (mut buffers, mut receipt) = population_fixture(73, max_terms);
            buffers.context.feature_count = 1924;
            buffers.buffers.context.feature_count = 1924;
            receipt.feature_count = 1924;
            for index in 0..73 {
                buffers.buffers.indices[index * max_terms as usize] = 1923;
            }
            let ffi = buffers.ffi_buffers_mut();
            assert_eq!(ffi.candidate_capacity, 73);
            assert_eq!(ffi.term_capacity, 73 * max_terms as u64);
            assert!(
                ffi.admission_sequences.is_null(),
                "population export must not allocate or copy archive sequence metadata"
            );
            let output = buffers.seal(receipt, &[314]).unwrap();
            assert_eq!(output.len(), 73);
            assert!(!output.is_empty());
            assert_eq!(output.max_terms(), max_terms);
            assert_eq!(output.feature_count(), 1924);
            assert_eq!(output.evaluated_generation(), 4);
            assert_eq!(output.host_copy_count(), 4);
            assert_eq!(output.host_copy_bytes(), 73 * (176 + 16 * max_terms as u64));
            for (index, candidate) in output.candidates().enumerate() {
                assert_eq!(candidate.gene_identity(), index as u64);
                assert_eq!(candidate.metric_row().candidate_id, index as u64);
                assert_eq!(candidate.generation(), (index % 5) as u32);
                assert_eq!(candidate.indices().len(), max_terms.min(2) as usize);
                assert_eq!(candidate.indices()[0], 1923);
                assert_eq!(candidate.weights()[0].to_bits(), (-0.0f64).to_bits());
            }
            assert!(output.candidate(73).is_none());
        }
    }

    #[test]
    fn terminal_population_keeps_economic_rejection_but_never_launders_metric_faults() {
        let (mut buffers, receipt) = population_fixture(2, 12);
        buffers.buffers.metrics[1].values[1] = f64::NEG_INFINITY;
        buffers.buffers.metrics[1].values[3] = 1.2;
        let output = buffers.seal(receipt, &[314]).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(
            output.candidate(1).unwrap().metric_row().values[1],
            f64::NEG_INFINITY
        );
        for slot in 0..11 {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let (mut buffers, receipt) = population_fixture(2, 12);
                buffers.buffers.metrics[1].values[slot] = invalid;
                // Even slot 1's -infinity is invalid with drawdown below one.
                assert!(matches!(
                    buffers.seal(receipt, &[314]),
                    Err(ResidentArchiveOutputErrorV3::Candidate { index: 1, .. })
                ));
            }
        }
        let (mut archive, receipt) = fixture(2);
        archive.metrics[1].values[1] = f64::NEG_INFINITY;
        archive.metrics[1].values[3] = 1.2;
        assert!(
            archive.seal(receipt, &[314]).is_err(),
            "archive admission must stay finite-only"
        );
    }

    #[test]
    fn terminal_population_refuses_stale_receipts_and_unevaluated_or_misbound_rows() {
        for field in 0..10 {
            let (buffers, mut receipt) = population_fixture(2, 12);
            match field {
                0 => receipt.abi_version += 1,
                1 => receipt.reserved = 1,
                2 => receipt.run_identity += 1,
                3 => receipt.packed_commit_word += 1,
                4 => receipt.evaluated_generation += 1,
                5 => receipt.candidate_count -= 1,
                6 => receipt.term_count -= 1,
                7 => receipt.feature_count += 1,
                8 => receipt.host_copy_count -= 1,
                _ => receipt.host_copy_bytes -= 1,
            }
            assert!(matches!(
                buffers.seal(receipt, &[314]),
                Err(ResidentArchiveOutputErrorV3::Authority(_))
            ));
        }
        for field in 0..6 {
            let (mut buffers, receipt) = population_fixture(2, 12);
            match field {
                0 => buffers.buffers.scalars[1].generation = 5,
                1 => buffers.buffers.scalars[1].term_count = 13,
                2 => buffers.buffers.metrics[1].candidate_id = 0,
                3 => buffers.buffers.metrics[1].scenario_id = 999,
                4 => buffers.buffers.indices[12] = 1024,
                _ => buffers.buffers.weights[14] = -0.0,
            }
            assert!(matches!(
                buffers.seal(receipt, &[314]),
                Err(ResidentArchiveOutputErrorV3::Candidate { index: 1, .. })
            ));
        }
        let (buffers, _) = population_fixture(1, 12);
        for context in [
            ResidentPopulationExportContextV3 {
                terminal_generation: 0,
                ..buffers.context
            },
            ResidentPopulationExportContextV3 {
                evaluated_generation: 5,
                ..buffers.context
            },
            ResidentPopulationExportContextV3 {
                candidate_count: 0,
                ..buffers.context
            },
            ResidentPopulationExportContextV3 {
                candidate_count: u64::MAX,
                ..buffers.context
            },
        ] {
            assert!(ResidentPopulationOutputBuffersV3::allocate(context).is_err());
        }
    }

    #[test]
    fn terminal_census_retains_population_even_when_archive_is_empty_or_tiny() {
        for archive_count in [0, 1] {
            let (archive, receipt) = fixture(archive_count);
            let archive = archive.seal(receipt, &[314]).unwrap();
            let (population, receipt) = population_fixture(73, 12);
            let population = population.seal(receipt, &[314]).unwrap();
            let census = ResidentSearchTerminalCandidatesV3::seal(archive, population).unwrap();
            assert_eq!(census.archive().len(), archive_count);
            assert_eq!(census.population().len(), 73);
            let (archive, population) = census.into_parts();
            assert_eq!(archive.len() + population.len(), archive_count + 73);
        }
        for mismatch in 0..5 {
            let (mut archive, mut receipt) = fixture(1);
            match mismatch {
                0 => {
                    archive.context.run_identity += 1;
                    receipt.run_identity += 1;
                }
                1 => {
                    archive.context.packed_commit_word += 1;
                    receipt.packed_commit_word += 1;
                }
                2 => {
                    archive.context.feature_count += 1;
                    receipt.feature_count += 1;
                }
                3 => archive.context.terminal_generation += 1,
                _ => archive.context.max_terms = 16,
            }
            let archive = archive.seal(receipt, &[314]).unwrap();
            let (population, receipt) = population_fixture(2, 12);
            let population = population.seal(receipt, &[314]).unwrap();
            assert!(ResidentSearchTerminalCandidatesV3::seal(archive, population).is_err());
        }
    }
}
