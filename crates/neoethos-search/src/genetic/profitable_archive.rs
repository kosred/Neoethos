//! Bounded retention of qualifying, evaluated GA observations.
//!
//! Admission thresholds remain at the generation caller. Retention uses the
//! existing archive handoff ranking: greater finite net profit wins, and an
//! equal net profit keeps the earlier admitted observation. This is not a
//! MAP-Elites or behavioral-novelty policy, and retained in-sample observations
//! are not independent OOS validation evidence.

use super::{EvaluatedGeneBehaviorKey, Gene};
use anyhow::{Result, anyhow};
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

type ArchiveObservation = (Gene, [f64; 11], usize);

/// The smallest item is the worst retained observation. At equal profit the
/// latest admission is worst; signed zero is one tie, as in the old comparator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArchiveRank {
    net_bits: u64,
    sequence: usize,
    slot: usize,
}

impl ArchiveRank {
    fn new(net: f64, sequence: usize, slot: usize) -> Self {
        debug_assert!(net.is_finite());
        Self {
            net_bits: if net == 0.0 { 0.0 } else { net }.to_bits(),
            sequence,
            slot,
        }
    }
}

impl Ord for ArchiveRank {
    fn cmp(&self, other: &Self) -> Ordering {
        f64::from_bits(self.net_bits)
            .total_cmp(&f64::from_bits(other.net_bits))
            .then_with(|| other.sequence.cmp(&self.sequence))
            .then_with(|| self.slot.cmp(&other.slot))
    }
}

impl PartialOrd for ArchiveRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArchiveOffer {
    Inserted,
    ReplacedWorst,
    ImprovedDuplicate,
    RejectedDuplicate,
    RejectedCapacity,
    RejectedNonFinite,
    RejectedNoTrades,
}

/// Exactly one observation, exact identity key and rank per retained slot.
/// Evictions remove both indexes; repeated improvements replace their rank in
/// place. No all-history key set or stale heap nodes can grow with generations.
pub(super) struct BoundedProfitableArchive {
    capacity: usize,
    observations: Vec<ArchiveObservation>,
    positions: HashMap<EvaluatedGeneBehaviorKey, usize>,
    ranks: BTreeSet<ArchiveRank>,
    next_sequence: usize,
}

impl BoundedProfitableArchive {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            observations: Vec::new(),
            positions: HashMap::new(),
            ranks: BTreeSet::new(),
            next_sequence: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.observations.len()
    }

    pub(super) fn offer(&mut self, gene: &Gene, metrics: &[f64; 11]) -> Result<ArchiveOffer> {
        // Retain the existing numeric/trade admission guard, including the
        // deliberate negative-infinite Sharpe marker for invalid account returns.
        if [metrics[0], metrics[1], metrics[5], metrics[8]]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Ok(ArchiveOffer::RejectedNonFinite);
        }
        if metrics[8] <= 0.0 {
            return Ok(ArchiveOffer::RejectedNoTrades);
        }
        if self.capacity == 0 {
            return Ok(ArchiveOffer::RejectedCapacity);
        }

        let key = EvaluatedGeneBehaviorKey::new(gene);
        if let Some(&slot) = self.positions.get(&key) {
            let incumbent = &self.observations[slot];
            if metrics[0] <= incumbent.1[0] {
                return Ok(ArchiveOffer::RejectedDuplicate);
            }
            let sequence = incumbent.2;
            let old_rank = ArchiveRank::new(incumbent.1[0], sequence, slot);
            self.ranks.remove(&old_rank);
            // Replace the whole observation: no new metrics attached to an old
            // gene ID, measured fitness, or other per-observation metadata.
            self.observations[slot] = (gene.clone(), *metrics, sequence);
            self.ranks
                .insert(ArchiveRank::new(metrics[0], sequence, slot));
            return Ok(ArchiveOffer::ImprovedDuplicate);
        }

        let worst = if self.observations.len() == self.capacity {
            let worst = *self
                .ranks
                .first()
                .expect("a nonempty full archive has a retained rank");
            if metrics[0] <= f64::from_bits(worst.net_bits) {
                return Ok(ArchiveOffer::RejectedCapacity);
            }
            Some(worst)
        } else {
            None
        };
        let sequence = self.next_sequence;
        let next_sequence = sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("GA archive admission sequence overflow"))?;

        let (slot, outcome) = if let Some(worst) = worst {
            let previous_key = EvaluatedGeneBehaviorKey::new(&self.observations[worst.slot].0);
            self.positions.remove(&previous_key);
            self.ranks.remove(&worst);
            self.observations[worst.slot] = (gene.clone(), *metrics, sequence);
            (worst.slot, ArchiveOffer::ReplacedWorst)
        } else {
            let slot = self.observations.len();
            self.observations.push((gene.clone(), *metrics, sequence));
            (slot, ArchiveOffer::Inserted)
        };
        self.positions.insert(key, slot);
        self.ranks
            .insert(ArchiveRank::new(metrics[0], sequence, slot));
        self.next_sequence = next_sequence;
        Ok(outcome)
    }

    pub(super) fn into_observations(self) -> Vec<ArchiveObservation> {
        self.observations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(index: usize, net: f64) -> (Gene, [f64; 11]) {
        let gene = Gene {
            indices: vec![index],
            weights: vec![0.4],
            long_threshold: 0.25,
            short_threshold: -0.25,
            sl_pips: 20.0,
            tp_pips: 40.0,
            strategy_id: format!("observation_{index}"),
            fitness: net,
            ..Gene::default()
        };
        let metrics = [
            net,
            1.0,
            10_000.0,
            0.1,
            0.6,
            1.2,
            net / 10.0,
            0.5,
            10.0,
            0.5,
            0.1,
        ];
        (gene, metrics)
    }

    fn offer(archive: &mut BoundedProfitableArchive, index: usize, net: f64) -> ArchiveOffer {
        let (gene, metrics) = observation(index, net);
        archive.offer(&gene, &metrics).expect("bounded observation")
    }

    fn retained_indices(archive: &BoundedProfitableArchive) -> Vec<usize> {
        let mut indices = archive
            .observations
            .iter()
            .map(|(gene, _, _)| gene.indices[0])
            .collect::<Vec<_>>();
        indices.sort_unstable();
        indices
    }

    fn assert_indexes(archive: &BoundedProfitableArchive) {
        assert!(archive.len() <= archive.capacity);
        assert_eq!(archive.positions.len(), archive.len());
        assert_eq!(archive.ranks.len(), archive.len());
        for (slot, (gene, metrics, sequence)) in archive.observations.iter().enumerate() {
            assert_eq!(
                archive.positions.get(&EvaluatedGeneBehaviorKey::new(gene)),
                Some(&slot)
            );
            assert!(
                archive
                    .ranks
                    .contains(&ArchiveRank::new(metrics[0], *sequence, slot))
            );
        }
    }

    #[test]
    fn a_full_archive_considers_later_better_worse_and_equal_candidates() {
        let mut archive = BoundedProfitableArchive::new(2);
        assert_eq!(offer(&mut archive, 0, 10.0), ArchiveOffer::Inserted);
        assert_eq!(offer(&mut archive, 1, 20.0), ArchiveOffer::Inserted);
        assert_eq!(offer(&mut archive, 2, 30.0), ArchiveOffer::ReplacedWorst);
        assert_eq!(offer(&mut archive, 3, 5.0), ArchiveOffer::RejectedCapacity);
        assert_eq!(offer(&mut archive, 4, 20.0), ArchiveOffer::RejectedCapacity);
        assert_eq!(retained_indices(&archive), vec![1, 2]);
        assert_indexes(&archive);
    }

    #[test]
    fn repeat_improvement_replaces_the_complete_aligned_observation_only_once() {
        let mut archive = BoundedProfitableArchive::new(2);
        offer(&mut archive, 0, 10.0);
        offer(&mut archive, 1, 20.0);
        let (mut updated, mut metrics) = observation(0, 30.0);
        updated.strategy_id = "later_measurement".into();
        updated.generation = 7;
        updated.fitness = 321.0;
        metrics[1] = 2.5;
        metrics[8] = 42.0;
        assert_eq!(
            archive.offer(&updated, &metrics).unwrap(),
            ArchiveOffer::ImprovedDuplicate
        );
        assert_eq!(
            offer(&mut archive, 0, 29.0),
            ArchiveOffer::RejectedDuplicate
        );
        assert_eq!(
            offer(&mut archive, 0, 30.0),
            ArchiveOffer::RejectedDuplicate
        );
        let stored = archive
            .observations
            .iter()
            .find(|(gene, _, _)| gene.indices == [0])
            .unwrap();
        assert_eq!(stored.0, updated);
        assert_eq!(stored.1, metrics);
        assert_eq!(
            stored.2, 0,
            "better repeat retains the original admission tie priority"
        );
        assert_indexes(&archive);
    }

    #[test]
    fn an_evicted_identity_can_reenter_with_a_later_better_observation() {
        let mut archive = BoundedProfitableArchive::new(2);
        offer(&mut archive, 0, 10.0);
        offer(&mut archive, 1, 20.0);
        offer(&mut archive, 2, 30.0);
        assert_eq!(offer(&mut archive, 0, 40.0), ArchiveOffer::ReplacedWorst);
        assert_eq!(retained_indices(&archive), vec![0, 2]);
        assert_indexes(&archive);
    }

    #[test]
    fn equal_net_ties_keep_earlier_admissions_even_when_slot_indices_are_reused() {
        let mut archive = BoundedProfitableArchive::new(3);
        for index in 0..3 {
            offer(&mut archive, index, 10.0);
        }
        offer(&mut archive, 3, 20.0);
        assert_eq!(retained_indices(&archive), vec![0, 1, 3]);
        offer(&mut archive, 1, 20.0);
        offer(&mut archive, 4, 30.0);
        offer(&mut archive, 5, 30.0);
        assert_eq!(retained_indices(&archive), vec![1, 4, 5]);
        assert_eq!(offer(&mut archive, 6, 20.0), ArchiveOffer::RejectedCapacity);
        assert_indexes(&archive);
    }

    #[test]
    fn exact_behavior_identity_does_not_merge_adjacent_bits_or_different_smc_flags() {
        let mut archive = BoundedProfitableArchive::new(4);
        let (base, metrics) = observation(0, 10.0);
        let mut adjacent = base.clone();
        adjacent.weights[0] = f64::from_bits(base.weights[0].to_bits() + 1);
        let mut smc = base.clone();
        smc.use_ob = !base.use_ob;
        assert_eq!(
            archive.offer(&base, &metrics).unwrap(),
            ArchiveOffer::Inserted
        );
        assert_eq!(
            archive.offer(&adjacent, &metrics).unwrap(),
            ArchiveOffer::Inserted
        );
        assert_eq!(
            archive.offer(&smc, &metrics).unwrap(),
            ArchiveOffer::Inserted
        );
        let mut renamed = base.clone();
        renamed.strategy_id = "not_a_new_behavior".into();
        assert_eq!(
            archive.offer(&renamed, &metrics).unwrap(),
            ArchiveOffer::RejectedDuplicate
        );
        assert_eq!(archive.len(), 3);
        assert_indexes(&archive);
    }

    #[test]
    fn finite_negative_net_is_ranked_without_a_new_zero_floor_and_nonfinite_is_rejected() {
        // The caller's existing `active` mode can admit negative net results.
        let mut archive = BoundedProfitableArchive::new(1);
        assert_eq!(offer(&mut archive, 0, -20.0), ArchiveOffer::Inserted);
        assert_eq!(offer(&mut archive, 1, -10.0), ArchiveOffer::ReplacedWorst);
        let (gene, good) = observation(2, 100.0);
        for slot in [0, 1, 5, 8] {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut metrics = good;
                metrics[slot] = invalid;
                assert_eq!(
                    archive.offer(&gene, &metrics).unwrap(),
                    ArchiveOffer::RejectedNonFinite
                );
            }
        }
        let mut no_trades = good;
        no_trades[8] = 0.0;
        assert_eq!(
            archive.offer(&gene, &no_trades).unwrap(),
            ArchiveOffer::RejectedNoTrades
        );
        assert_eq!(retained_indices(&archive), vec![1]);
        assert_indexes(&archive);
    }

    #[test]
    fn signed_zero_profit_is_a_stable_tie_not_a_replacement() {
        let mut archive = BoundedProfitableArchive::new(1);
        offer(&mut archive, 0, -0.0);
        assert_eq!(offer(&mut archive, 1, 0.0), ArchiveOffer::RejectedCapacity);
        assert_eq!(retained_indices(&archive), vec![0]);
        assert_indexes(&archive);
    }

    #[test]
    fn signed_zero_handoff_preserves_the_same_admission_tie_order() {
        let mut archive = BoundedProfitableArchive::new(2);
        offer(&mut archive, 0, -0.0);
        offer(&mut archive, 1, 0.0);
        let result = super::super::finish_evaluated_generation(
            archive.into_observations(),
            Vec::new(),
            0.35,
            2,
        );
        assert_eq!(
            result
                .genes
                .iter()
                .map(|gene| gene.indices[0])
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(result.metrics[0][0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(result.metrics[1][0].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn storage_and_both_indexes_stay_bounded_across_repeated_improvements_and_evictions() {
        let mut archive = BoundedProfitableArchive::new(16);
        for observation_number in 0..10_000 {
            let index = observation_number % 32;
            offer(&mut archive, index, observation_number as f64);
            // Exercise same-identity improvements too: a stale-heap design
            // could grow without increasing its reported archive length.
            offer(&mut archive, index, observation_number as f64 + 0.5);
            assert_indexes(&archive);
        }
        assert_eq!(archive.len(), 16);
        let zero = BoundedProfitableArchive::new(0);
        assert_indexes(&zero);
        let mut zero = zero;
        assert_eq!(offer(&mut zero, 0, 1.0), ArchiveOffer::RejectedCapacity);
        assert_indexes(&zero);
    }

    #[test]
    fn final_population_still_overrides_the_retained_observation_with_its_exact_metrics() {
        let mut archive = BoundedProfitableArchive::new(1);
        offer(&mut archive, 0, 10.0);
        let (mut current, current_metrics) = observation(0, -5.0);
        current.strategy_id = "current_final_gate_measurement".into();
        let result = super::super::finish_evaluated_generation(
            archive.into_observations(),
            vec![(current.fitness, 0, current.clone(), current_metrics)],
            0.35,
            2,
        );
        assert_eq!(result.genes, vec![current]);
        assert_eq!(result.metrics, vec![current_metrics]);
    }
}
