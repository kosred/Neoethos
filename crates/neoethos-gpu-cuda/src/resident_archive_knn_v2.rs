//! Exact host-side archive and k-nearest-neighbor contract for resident search.

use std::cmp::Ordering;

/// Number of `u64` words in one novelty signature.
pub const SIGNATURE_WORDS: usize = 4;
/// Maximum number of neighbors contributing to one novelty mean.
pub const K_NEIGHBORS: usize = 15;
/// Wire discriminator for a member of the current population.
pub const SOURCE_CURRENT: u8 = 0;
/// Wire discriminator for a committed archive member.
pub const SOURCE_ARCHIVE: u8 = 1;
/// Number of metrics carried by an archive admission.
pub const METRIC_COUNT: usize = 11;
/// Metric slot used by the strictly-positive net gate.
pub const NET_METRIC_SLOT: usize = 0;
/// Metric slot used by the strictly-positive trade-count gate.
pub const TRADE_COUNT_METRIC_SLOT: usize = 8;

const MAX_EXACT_F64_INTEGER: u128 = 1_u128 << 53;

/// One current-population or committed-archive signature presented to kNN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborWire {
    pub source_kind: u8,
    pub source_ordinal: u32,
    pub gene_identity: u64,
    pub signature: [u64; SIGNATURE_WORDS],
}

/// Unreduced Jaccard-distance fraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistanceFraction {
    pub numerator: u32,
    pub denominator: u32,
}

impl DistanceFraction {
    /// Constructs a fraction without reducing it.
    pub const fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}

/// A selected neighbor in canonical total order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectedNeighbor {
    pub source_kind: u8,
    pub source_ordinal: u32,
    pub gene_identity: u64,
    pub distance: DistanceFraction,
}

/// Checked rational used for exact comparison and summation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedRational {
    pub numerator: u128,
    pub denominator: u128,
}

impl CheckedRational {
    /// Constructs a rational; consumers validate a nonzero denominator.
    pub const fn new(numerator: u128, denominator: u128) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}

/// Complete exact-kNN selection and its sealed floating-point mean.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnResult {
    pub selected: Vec<SelectedNeighbor>,
    pub exact_sum: CheckedRational,
    pub mean: f64,
}

/// Fail-closed validation and arithmetic failures from exact kNN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnError {
    InvalidSourceKind {
        wire: u8,
    },
    CurrentOrdinalOutOfDomain {
        ordinal: u32,
        population_count: u32,
    },
    ArchiveOrdinalOutOfDomain {
        ordinal: u32,
        archive_count: u32,
    },
    ZeroUnion {
        source_kind: u8,
        source_ordinal: u32,
    },
    DenominatorOutOfDomain {
        denominator: u32,
    },
    NumeratorExceedsDenominator {
        numerator: u32,
        denominator: u32,
    },
    ZeroAvailableNeighbors,
    ComparatorOverflow,
    RationalZeroDenominator,
    RationalSumOverflow,
    IntegerNotExactlyRepresentable {
        value: u128,
    },
}

/// Validates the exact resident-signature distance domain.
pub fn validate_distance_fraction(
    numerator: u32,
    denominator: u32,
) -> Result<DistanceFraction, KnnError> {
    if !(1..=32).contains(&denominator) {
        return Err(KnnError::DenominatorOutOfDomain { denominator });
    }
    if numerator > denominator {
        return Err(KnnError::NumeratorExceedsDenominator {
            numerator,
            denominator,
        });
    }
    Ok(DistanceFraction::new(numerator, denominator))
}

/// Compares two rationals with checked cross-products.
pub fn compare_fraction_checked(
    left: CheckedRational,
    right: CheckedRational,
) -> Result<Ordering, KnnError> {
    if left.denominator == 0 || right.denominator == 0 {
        return Err(KnnError::RationalZeroDenominator);
    }
    let left_cross = left
        .numerator
        .checked_mul(right.denominator)
        .ok_or(KnnError::ComparatorOverflow)?;
    let right_cross = right
        .numerator
        .checked_mul(left.denominator)
        .ok_or(KnnError::ComparatorOverflow)?;
    Ok(left_cross.cmp(&right_cross))
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn reduce_rational(value: CheckedRational) -> Result<CheckedRational, KnnError> {
    if value.denominator == 0 {
        return Err(KnnError::RationalZeroDenominator);
    }
    let divisor = gcd_u128(value.numerator, value.denominator);
    Ok(CheckedRational::new(
        value.numerator / divisor,
        value.denominator / divisor,
    ))
}

fn add_rational_checked(
    left: CheckedRational,
    right: CheckedRational,
) -> Result<CheckedRational, KnnError> {
    if left.denominator == 0 || right.denominator == 0 {
        return Err(KnnError::RationalZeroDenominator);
    }
    let denominator_gcd = gcd_u128(left.denominator, right.denominator);
    let left_scale = right.denominator / denominator_gcd;
    let right_scale = left.denominator / denominator_gcd;
    let left_numerator = left
        .numerator
        .checked_mul(left_scale)
        .ok_or(KnnError::RationalSumOverflow)?;
    let right_numerator = right
        .numerator
        .checked_mul(right_scale)
        .ok_or(KnnError::RationalSumOverflow)?;
    let numerator = left_numerator
        .checked_add(right_numerator)
        .ok_or(KnnError::RationalSumOverflow)?;
    let denominator = left
        .denominator
        .checked_mul(left_scale)
        .ok_or(KnnError::RationalSumOverflow)?;
    reduce_rational(CheckedRational::new(numerator, denominator))
}

/// Sums rationals exactly and returns the reduced result.
pub fn checked_rational_sum(fractions: &[CheckedRational]) -> Result<CheckedRational, KnnError> {
    let mut sum = CheckedRational::new(0, 1);
    for &fraction in fractions {
        sum = add_rational_checked(sum, fraction)?;
    }
    Ok(sum)
}

fn distance_fraction(
    query_signature: [u64; SIGNATURE_WORDS],
    candidate: NeighborWire,
) -> Result<DistanceFraction, KnnError> {
    let intersection = query_signature
        .iter()
        .zip(candidate.signature.iter())
        .map(|(&query_word, &candidate_word)| (query_word & candidate_word).count_ones())
        .sum::<u32>();
    let union = query_signature
        .iter()
        .zip(candidate.signature.iter())
        .map(|(&query_word, &candidate_word)| (query_word | candidate_word).count_ones())
        .sum::<u32>();
    if union == 0 {
        return Err(KnnError::ZeroUnion {
            source_kind: candidate.source_kind,
            source_ordinal: candidate.source_ordinal,
        });
    }
    validate_distance_fraction(union - intersection, union)
}

fn validate_wire_domain(
    candidate: NeighborWire,
    population_count: u32,
    archive_count: u32,
) -> Result<(), KnnError> {
    match candidate.source_kind {
        SOURCE_CURRENT if candidate.source_ordinal < population_count => Ok(()),
        SOURCE_CURRENT => Err(KnnError::CurrentOrdinalOutOfDomain {
            ordinal: candidate.source_ordinal,
            population_count,
        }),
        SOURCE_ARCHIVE if candidate.source_ordinal < archive_count => Ok(()),
        SOURCE_ARCHIVE => Err(KnnError::ArchiveOrdinalOutOfDomain {
            ordinal: candidate.source_ordinal,
            archive_count,
        }),
        wire => Err(KnnError::InvalidSourceKind { wire }),
    }
}

fn compare_selected(
    left: &SelectedNeighbor,
    right: &SelectedNeighbor,
) -> Result<Ordering, KnnError> {
    let rational_order = compare_fraction_checked(
        CheckedRational::new(
            u128::from(left.distance.numerator),
            u128::from(left.distance.denominator),
        ),
        CheckedRational::new(
            u128::from(right.distance.numerator),
            u128::from(right.distance.denominator),
        ),
    )?;
    Ok(rational_order
        .then_with(|| left.gene_identity.cmp(&right.gene_identity))
        .then_with(|| left.source_kind.cmp(&right.source_kind))
        .then_with(|| left.source_ordinal.cmp(&right.source_ordinal)))
}

fn sort_selected_checked(neighbors: &mut [SelectedNeighbor]) -> Result<(), KnnError> {
    let mut comparator_error = None;
    neighbors.sort_by(|left, right| {
        if comparator_error.is_some() {
            return Ordering::Equal;
        }
        match compare_selected(left, right) {
            Ok(ordering) => ordering,
            Err(error) => {
                comparator_error = Some(error);
                Ordering::Equal
            }
        }
    });
    comparator_error.map_or(Ok(()), Err)
}

fn exact_u128_to_f64(value: u128) -> Result<f64, KnnError> {
    if value > MAX_EXACT_F64_INTEGER {
        return Err(KnnError::IntegerNotExactlyRepresentable { value });
    }
    Ok(value as f64)
}

fn sealed_mean(selected: &[SelectedNeighbor]) -> Result<f64, KnnError> {
    let mut sum = 0.0_f64;
    for neighbor in selected {
        let numerator = exact_u128_to_f64(u128::from(neighbor.distance.numerator))?;
        let denominator = exact_u128_to_f64(u128::from(neighbor.distance.denominator))?;
        sum += numerator / denominator;
    }
    Ok(sum / exact_u128_to_f64(selected.len() as u128)?)
}

/// Selects at most 15 exact Jaccard neighbors in canonical total order.
pub fn select_exact_knn(
    query_current_ordinal: u32,
    query_signature: [u64; SIGNATURE_WORDS],
    population_count: u32,
    archive_count: u32,
    neighbors: &[NeighborWire],
) -> Result<KnnResult, KnnError> {
    if query_current_ordinal >= population_count {
        return Err(KnnError::CurrentOrdinalOutOfDomain {
            ordinal: query_current_ordinal,
            population_count,
        });
    }

    let mut selected = Vec::with_capacity(neighbors.len());
    for &candidate in neighbors {
        validate_wire_domain(candidate, population_count, archive_count)?;
        if candidate.source_kind == SOURCE_CURRENT
            && candidate.source_ordinal == query_current_ordinal
        {
            continue;
        }
        selected.push(SelectedNeighbor {
            source_kind: candidate.source_kind,
            source_ordinal: candidate.source_ordinal,
            gene_identity: candidate.gene_identity,
            distance: distance_fraction(query_signature, candidate)?,
        });
    }
    if selected.is_empty() {
        return Err(KnnError::ZeroAvailableNeighbors);
    }

    sort_selected_checked(&mut selected)?;
    selected.truncate(K_NEIGHBORS.min(selected.len()));
    let fractions = selected
        .iter()
        .map(|neighbor| {
            CheckedRational::new(
                u128::from(neighbor.distance.numerator),
                u128::from(neighbor.distance.denominator),
            )
        })
        .collect::<Vec<_>>();
    let exact_sum = checked_rational_sum(&fractions)?;
    let mean = sealed_mean(&selected)?;
    Ok(KnnResult {
        selected,
        exact_sum,
        mean,
    })
}

fn nonnegative_finite_ulp_distance(left: f64, right: f64) -> Option<u64> {
    if !left.is_finite()
        || !right.is_finite()
        || left.to_bits() >> 63 != 0
        || right.to_bits() >> 63 != 0
    {
        return None;
    }
    Some(left.to_bits().abs_diff(right.to_bits()))
}

/// Checks the absolute, relative, and ULP bounds of a sealed novelty mean.
pub fn accept_sealed_mean(expected: f64, actual: f64) -> bool {
    let Some(ulp_distance) = nonnegative_finite_ulp_distance(expected, actual) else {
        return false;
    };
    if expected.to_bits() == 0 {
        return actual.to_bits() == 0;
    }
    let absolute_error = (actual - expected).abs();
    let relative_error = absolute_error / expected;
    absolute_error <= 2.0_f64.powi(-50) && relative_error <= 2.0_f64.powi(-48) && ulp_distance <= 4
}

/// Canonical normalized full-gene value used for exact deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactGene(pub [u64; 2]);

/// Ranked candidate presented to the archive admission stage.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionCandidate {
    pub exact_gene: ExactGene,
    pub full_gene_hash: u64,
    pub gene_identity: u64,
    pub population_ordinal: u32,
    pub score: f64,
    pub metrics: [f64; METRIC_COUNT],
}

/// Immutable committed archive payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveRecord {
    pub exact_gene: ExactGene,
    pub full_gene_hash: u64,
    pub gene_identity: u64,
    pub population_ordinal: u32,
    pub admitted_generation: u32,
    pub metric_bits: [u64; METRIC_COUNT],
}

/// Generation-start view containing committed records only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborSnapshot {
    pub generation: u32,
    pub records: Vec<ArchiveRecord>,
}

/// Prepared but not yet published archive tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageReceipt {
    pub generation: u32,
    pub committed_count_at_start: usize,
    pub target_committed_count: usize,
    pub full_gene_hash_collision_count: u64,
    pub records: Vec<ArchiveRecord>,
}

/// Atomic publication receipt for one completed generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitReceipt {
    pub completed_generation: u32,
    pub next_generation: u32,
    pub previous_committed_count: usize,
    pub committed_count: usize,
}

/// Fail-closed archive transaction error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveError {
    GenerationMismatch {
        current: u32,
        requested: u32,
    },
    StageAlreadyPrepared {
        generation: u32,
    },
    TransactionFaulted {
        generation: u32,
    },
    NonFiniteMetric {
        population_ordinal: u32,
        metric_slot: usize,
    },
    NoStagedTransaction {
        generation: u32,
    },
    InitialArchiveExceedsCapacity {
        capacity: usize,
        record_count: usize,
    },
    GenerationOverflow {
        generation: u32,
    },
    CollisionCountOverflow {
        generation: u32,
    },
}

#[derive(Debug)]
struct PendingArchive {
    generation: u32,
    target_full_gene_hash_collision_count: u64,
    records: Vec<ArchiveRecord>,
}

/// Two-phase exact-gene archive with generation-atomic publication.
#[derive(Debug)]
pub struct ResidentArchiveKnnV2 {
    capacity: usize,
    current_generation: u32,
    committed: Vec<ArchiveRecord>,
    committed_full_gene_hash_collision_count: u64,
    pending: Option<PendingArchive>,
    faulted_generation: Option<u32>,
}

impl ResidentArchiveKnnV2 {
    /// Creates an empty generation-zero archive.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            current_generation: 0,
            committed: Vec::new(),
            committed_full_gene_hash_collision_count: 0,
            pending: None,
            faulted_generation: None,
        }
    }

    /// Restores an already committed archive without creating a staged tail.
    pub fn from_committed(
        capacity: usize,
        current_generation: u32,
        committed: Vec<ArchiveRecord>,
        full_gene_hash_collision_count: u64,
    ) -> Result<Self, ArchiveError> {
        if committed.len() > capacity {
            return Err(ArchiveError::InitialArchiveExceedsCapacity {
                capacity,
                record_count: committed.len(),
            });
        }
        Ok(Self {
            capacity,
            current_generation,
            committed,
            committed_full_gene_hash_collision_count: full_gene_hash_collision_count,
            pending: None,
            faulted_generation: None,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn current_generation(&self) -> u32 {
        self.current_generation
    }

    pub fn committed_count(&self) -> usize {
        self.committed.len()
    }

    pub fn committed_records(&self) -> &[ArchiveRecord] {
        &self.committed
    }

    pub fn committed_full_gene_hash_collision_count(&self) -> u64 {
        self.committed_full_gene_hash_collision_count
    }

    pub fn staged_count(&self) -> usize {
        self.pending
            .as_ref()
            .map_or(0, |pending| pending.records.len())
    }

    pub fn faulted_generation(&self) -> Option<u32> {
        self.faulted_generation
    }

    /// Clones the committed-only view for the requested current generation.
    pub fn neighbor_snapshot_at_generation_start(
        &self,
        generation: u32,
    ) -> Result<NeighborSnapshot, ArchiveError> {
        self.require_current_generation(generation)?;
        Ok(NeighborSnapshot {
            generation,
            records: self.committed.clone(),
        })
    }

    /// Validates and prepares ranked admissions without publishing them.
    pub fn stage_ranked_admissions(
        &mut self,
        generation: u32,
        candidates: &[AdmissionCandidate],
    ) -> Result<StageReceipt, ArchiveError> {
        self.require_current_generation(generation)?;
        if let Some(faulted_generation) = self.faulted_generation {
            return Err(ArchiveError::TransactionFaulted {
                generation: faulted_generation,
            });
        }
        if self.pending.is_some() {
            return Err(ArchiveError::StageAlreadyPrepared { generation });
        }

        for candidate in candidates {
            if let Some(metric_slot) = candidate
                .metrics
                .iter()
                .position(|metric| !metric.is_finite())
            {
                self.faulted_generation = Some(generation);
                return Err(ArchiveError::NonFiniteMetric {
                    population_ordinal: candidate.population_ordinal,
                    metric_slot,
                });
            }
        }

        let mut ranked = candidates.iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| compare_rank(left, right));
        let available_slots = self.capacity.saturating_sub(self.committed.len());
        let mut target_collision_count = self.committed_full_gene_hash_collision_count;
        let mut staged = Vec::with_capacity(available_slots.min(ranked.len()));

        for candidate in ranked {
            if candidate.metrics[TRADE_COUNT_METRIC_SLOT] <= 0.0
                || candidate.metrics[NET_METRIC_SLOT] <= 0.0
            {
                continue;
            }
            if self
                .committed
                .iter()
                .chain(staged.iter())
                .any(|record| record.exact_gene == candidate.exact_gene)
            {
                continue;
            }
            if staged.len() == available_slots {
                break;
            }

            let same_hash_unequal_gene_count = self
                .committed
                .iter()
                .chain(staged.iter())
                .filter(|record| {
                    record.full_gene_hash == candidate.full_gene_hash
                        && record.exact_gene != candidate.exact_gene
                })
                .try_fold(0_u64, |count, _| count.checked_add(1))
                .ok_or_else(|| {
                    self.faulted_generation = Some(generation);
                    ArchiveError::CollisionCountOverflow { generation }
                })?;
            target_collision_count = target_collision_count
                .checked_add(same_hash_unequal_gene_count)
                .ok_or_else(|| {
                    self.faulted_generation = Some(generation);
                    ArchiveError::CollisionCountOverflow { generation }
                })?;

            staged.push(ArchiveRecord {
                exact_gene: candidate.exact_gene,
                full_gene_hash: candidate.full_gene_hash,
                gene_identity: candidate.gene_identity,
                population_ordinal: candidate.population_ordinal,
                admitted_generation: generation,
                metric_bits: candidate.metrics.map(f64::to_bits),
            });
        }

        let committed_count_at_start = self.committed.len();
        let target_committed_count = committed_count_at_start + staged.len();
        let receipt = StageReceipt {
            generation,
            committed_count_at_start,
            target_committed_count,
            full_gene_hash_collision_count: target_collision_count,
            records: staged.clone(),
        };
        self.pending = Some(PendingArchive {
            generation,
            target_full_gene_hash_collision_count: target_collision_count,
            records: staged,
        });
        Ok(receipt)
    }

    /// Poisons an existing staged transaction before publication.
    pub fn fault_staged_transaction_before_commit(
        &mut self,
        generation: u32,
    ) -> Result<(), ArchiveError> {
        self.require_current_generation(generation)?;
        if let Some(faulted_generation) = self.faulted_generation {
            return Err(ArchiveError::TransactionFaulted {
                generation: faulted_generation,
            });
        }
        let pending = self
            .pending
            .as_ref()
            .ok_or(ArchiveError::NoStagedTransaction { generation })?;
        if pending.generation != generation {
            return Err(ArchiveError::GenerationMismatch {
                current: pending.generation,
                requested: generation,
            });
        }
        self.faulted_generation = Some(generation);
        Ok(())
    }

    /// Atomically publishes the prepared tail and advances one generation.
    pub fn combined_commit(&mut self, generation: u32) -> Result<CommitReceipt, ArchiveError> {
        self.require_current_generation(generation)?;
        if let Some(faulted_generation) = self.faulted_generation {
            return Err(ArchiveError::TransactionFaulted {
                generation: faulted_generation,
            });
        }
        let pending = self
            .pending
            .as_ref()
            .ok_or(ArchiveError::NoStagedTransaction { generation })?;
        if pending.generation != generation {
            return Err(ArchiveError::GenerationMismatch {
                current: pending.generation,
                requested: generation,
            });
        }
        let next_generation = generation
            .checked_add(1)
            .ok_or(ArchiveError::GenerationOverflow { generation })?;
        let previous_committed_count = self.committed.len();
        let pending = self
            .pending
            .take()
            .ok_or(ArchiveError::NoStagedTransaction { generation })?;
        self.committed.extend(pending.records);
        self.committed_full_gene_hash_collision_count =
            pending.target_full_gene_hash_collision_count;
        self.current_generation = next_generation;
        Ok(CommitReceipt {
            completed_generation: generation,
            next_generation,
            previous_committed_count,
            committed_count: self.committed.len(),
        })
    }

    fn require_current_generation(&self, requested: u32) -> Result<(), ArchiveError> {
        if requested != self.current_generation {
            return Err(ArchiveError::GenerationMismatch {
                current: self.current_generation,
                requested,
            });
        }
        Ok(())
    }
}

/// Canonical total rank: score descending, identity ascending, ordinal ascending.
pub fn compare_rank(left: &AdmissionCandidate, right: &AdmissionCandidate) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.gene_identity.cmp(&right.gene_identity))
        .then_with(|| left.population_ordinal.cmp(&right.population_ordinal))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature(word0: u64) -> [u64; SIGNATURE_WORDS] {
        [word0, 0, 0, 0]
    }

    fn candidate(
        gene: [u64; 2],
        hash: u64,
        identity: u64,
        ordinal: u32,
        score: f64,
        net: f64,
        trades: f64,
    ) -> AdmissionCandidate {
        let mut metrics = [1.0; METRIC_COUNT];
        metrics[NET_METRIC_SLOT] = net;
        metrics[TRADE_COUNT_METRIC_SLOT] = trades;
        AdmissionCandidate {
            exact_gene: ExactGene(gene),
            full_gene_hash: hash,
            gene_identity: identity,
            population_ordinal: ordinal,
            score,
            metrics,
        }
    }

    #[test]
    fn exact_knn_uses_rational_order_self_exclusion_and_sealed_mean() {
        let query = signature(0b0011);
        let neighbors = [
            NeighborWire {
                source_kind: SOURCE_CURRENT,
                source_ordinal: 0,
                gene_identity: 99,
                signature: query,
            },
            NeighborWire {
                source_kind: SOURCE_ARCHIVE,
                source_ordinal: 0,
                gene_identity: 20,
                signature: signature(0b1111),
            },
            NeighborWire {
                source_kind: SOURCE_CURRENT,
                source_ordinal: 1,
                gene_identity: 30,
                signature: signature(0b0001),
            },
            NeighborWire {
                source_kind: SOURCE_CURRENT,
                source_ordinal: 2,
                gene_identity: 40,
                signature: signature(0b1_1111),
            },
        ];

        let result = select_exact_knn(0, query, 3, 1, &neighbors).unwrap();
        assert_eq!(result.selected.len(), 3);
        assert_eq!(result.selected[0].gene_identity, 20);
        assert_eq!(result.selected[0].distance, DistanceFraction::new(2, 4));
        assert_eq!(result.selected[1].gene_identity, 30);
        assert_eq!(result.selected[1].distance, DistanceFraction::new(1, 2));
        assert_eq!(result.selected[2].gene_identity, 40);
        assert_eq!(result.selected[2].distance, DistanceFraction::new(3, 5));
        assert_eq!(result.exact_sum, CheckedRational::new(8, 5));
        assert_eq!(
            result.mean.to_bits(),
            ((0.5 + 0.5 + 0.6) / 3.0_f64).to_bits()
        );
        assert!(accept_sealed_mean(result.mean, result.mean));
    }

    #[test]
    fn exact_knn_rejects_invalid_domains_and_zero_union() {
        assert_eq!(
            validate_distance_fraction(1, 0),
            Err(KnnError::DenominatorOutOfDomain { denominator: 0 })
        );
        assert_eq!(
            validate_distance_fraction(1, 33),
            Err(KnnError::DenominatorOutOfDomain { denominator: 33 })
        );
        assert_eq!(
            select_exact_knn(
                0,
                signature(0),
                2,
                0,
                &[
                    NeighborWire {
                        source_kind: SOURCE_CURRENT,
                        source_ordinal: 0,
                        gene_identity: 1,
                        signature: signature(0),
                    },
                    NeighborWire {
                        source_kind: SOURCE_CURRENT,
                        source_ordinal: 1,
                        gene_identity: 2,
                        signature: signature(0),
                    },
                ],
            ),
            Err(KnnError::ZeroUnion {
                source_kind: SOURCE_CURRENT,
                source_ordinal: 1,
            })
        );
    }

    #[test]
    fn archive_stage_is_invisible_until_commit_and_deduplicates_exact_genes() {
        let mut archive = ResidentArchiveKnnV2::new(3);
        let first = candidate([1, 10], 7, 20, 2, 10.0, 2.0, 3.0);
        let same_hash_other_gene = candidate([2, 20], 7, 30, 1, 9.0, 4.0, 5.0);
        let later_duplicate = candidate([2, 20], 7, 1, 0, 8.0, 6.0, 7.0);

        let staged = archive
            .stage_ranked_admissions(0, &[later_duplicate, same_hash_other_gene, first])
            .unwrap();
        assert_eq!(archive.committed_count(), 0);
        assert_eq!(archive.staged_count(), 2);
        assert_eq!(staged.records[0].exact_gene, ExactGene([1, 10]));
        assert_eq!(staged.records[1].exact_gene, ExactGene([2, 20]));
        assert_eq!(staged.full_gene_hash_collision_count, 1);
        assert_eq!(
            archive
                .neighbor_snapshot_at_generation_start(0)
                .unwrap()
                .records,
            vec![]
        );

        let receipt = archive.combined_commit(0).unwrap();
        assert_eq!(receipt.next_generation, 1);
        assert_eq!(archive.committed_count(), 2);
        assert_eq!(archive.committed_full_gene_hash_collision_count(), 1);
        assert_eq!(
            archive.committed_records()[1].metric_bits[NET_METRIC_SLOT],
            4.0_f64.to_bits()
        );
    }

    #[test]
    fn archive_nonfinite_metric_faults_atomically() {
        let mut archive = ResidentArchiveKnnV2::new(2);
        let mut invalid = candidate([9, 90], 9, 9, 4, 1.0, 1.0, 1.0);
        invalid.metrics[5] = f64::NAN;

        assert_eq!(
            archive.stage_ranked_admissions(0, &[invalid]),
            Err(ArchiveError::NonFiniteMetric {
                population_ordinal: 4,
                metric_slot: 5,
            })
        );
        assert_eq!(archive.committed_count(), 0);
        assert_eq!(archive.staged_count(), 0);
        assert_eq!(archive.faulted_generation(), Some(0));
        assert_eq!(
            archive.combined_commit(0),
            Err(ArchiveError::TransactionFaulted { generation: 0 })
        );
    }
}
