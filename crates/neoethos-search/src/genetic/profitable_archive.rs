//! Bounded retention of qualifying, evaluated GA observations.
//!
//! Admission thresholds remain at the generation caller. Retention uses the
//! legacy archive handoff ranking: greater finite net profit wins. Explicit
//! weekly runs retain the measured feasibility fitness first instead. Equal
//! rank keeps the earlier admitted observation. This is not a
//! MAP-Elites or behavioral-novelty policy, and retained in-sample observations
//! are not independent OOS validation evidence.

use super::{EvaluatedGeneBehaviorKey, Gene};
use crate::scoring::weekly_cadence::{
    WeeklyRollingCadenceReportV1, WeeklySearchObjectiveV1, weekly_select_survivors,
    weekly_selection_ranks,
};
use anyhow::{Result, anyhow};
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

type ArchiveObservation = (Gene, [f64; 11], usize);

/// The smallest item is the worst retained observation. At equal profit the
/// latest admission is worst; signed zero is one tie, as in the old comparator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArchiveRank {
    score_bits: u64,
    sequence: usize,
    slot: usize,
}

impl ArchiveRank {
    fn new(value: f64, sequence: usize, slot: usize) -> Self {
        debug_assert!(value.is_finite());
        Self {
            score_bits: if value == 0.0 { 0.0 } else { value }.to_bits(),
            sequence,
            slot,
        }
    }
}

impl Ord for ArchiveRank {
    fn cmp(&self, other: &Self) -> Ordering {
        f64::from_bits(self.score_bits)
            .total_cmp(&f64::from_bits(other.score_bits))
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

/// Bounded, versioned research retention set for weekly vector-front selection.
/// The 1024 ceiling is explicit policy, not a hidden cap: all admitted
/// candidates are still evaluated by the caller, which reports counts.
pub(super) const WEEKLY_VECTOR_ARCHIVE_CAPACITY_V1: usize = 1024;

#[derive(Clone, Debug)]
pub(super) struct WeeklyVectorObservationV1 {
    pub gene: Gene,
    pub metrics: [f64; 11],
    pub report: WeeklyRollingCadenceReportV1,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct WeeklyArchiveSummaryV1 {
    pub capacity: usize,
    pub evaluated: usize,
    pub retained: usize,
    pub capacity_rejected: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct WeeklyArchiveBatchCensusV1 {
    pub evaluated: usize,
    pub distinct: usize,
    pub retained: usize,
    pub capacity_rejected: usize,
    pub invalid: usize,
}

/// Exactly one observation, exact identity key and rank per retained slot.
/// Evictions remove both indexes; repeated improvements replace their rank in
/// place. No all-history key set or stale heap nodes can grow with generations.
pub(super) struct BoundedProfitableArchive {
    capacity: usize,
    weekly_fitness: bool,
    weekly_vector_capacity: usize,
    weekly_observations: Vec<WeeklyVectorObservationV1>,
    weekly_evaluated: usize,
    weekly_capacity_rejected: usize,
    observations: Vec<ArchiveObservation>,
    positions: HashMap<EvaluatedGeneBehaviorKey, usize>,
    ranks: BTreeSet<ArchiveRank>,
    next_sequence: usize,
}

impl BoundedProfitableArchive {
    #[cfg(test)]
    pub(super) fn new(capacity: usize) -> Self {
        Self::for_objective(capacity, false)
    }

    pub(super) fn for_objective(capacity: usize, weekly_fitness: bool) -> Self {
        Self {
            weekly_fitness,
            weekly_vector_capacity: 0,
            weekly_observations: Vec::new(),
            weekly_evaluated: 0,
            weekly_capacity_rejected: 0,
            capacity,
            observations: Vec::new(),
            positions: HashMap::new(),
            ranks: BTreeSet::new(),
            next_sequence: 0,
        }
    }

    /// Bounded vector-front retention for weekly runs: one union re-rank per
    /// admitted batch with the explicit 1024 research working-set ceiling.
    pub(super) fn for_weekly_vector_objective(capacity: usize) -> Self {
        let mut archive = Self::for_objective(capacity, false);
        archive.weekly_vector_capacity = capacity.min(WEEKLY_VECTOR_ARCHIVE_CAPACITY_V1);
        archive
    }

    pub(super) fn resolved_weekly_vector_capacity(&self) -> usize {
        self.weekly_vector_capacity
    }

    #[cfg(test)]
    pub(super) fn weekly_vector_len(&self) -> usize {
        self.weekly_observations.len()
    }

    #[cfg(test)]
    pub(super) fn weekly_vector_evaluated(&self) -> usize {
        self.weekly_evaluated
    }

    #[cfg(test)]
    pub(super) fn weekly_vector_capacity_rejected(&self) -> usize {
        self.weekly_capacity_rejected
    }

    #[cfg(test)]
    pub(super) fn weekly_observations(&self) -> &[WeeklyVectorObservationV1] {
        &self.weekly_observations
    }

    fn retention_value(&self, gene: &Gene, metrics: &[f64; 11]) -> f64 {
        if self.weekly_fitness {
            gene.fitness
        } else {
            metrics[0]
        }
    }

    pub(super) fn len(&self) -> usize {
        if self.weekly_vector_capacity > 0 {
            self.weekly_observations.len()
        } else {
            self.observations.len()
        }
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

        let value = self.retention_value(gene, metrics);
        if !value.is_finite() {
            return Ok(ArchiveOffer::RejectedNonFinite);
        }
        let key = EvaluatedGeneBehaviorKey::new(gene);
        if let Some(&slot) = self.positions.get(&key) {
            let incumbent = &self.observations[slot];
            let incumbent_value = self.retention_value(&incumbent.0, &incumbent.1);
            if value <= incumbent_value {
                return Ok(ArchiveOffer::RejectedDuplicate);
            }
            let sequence = incumbent.2;
            let old_rank = ArchiveRank::new(incumbent_value, sequence, slot);
            self.ranks.remove(&old_rank);
            // Replace the whole observation: no new metrics attached to an old
            // gene ID, measured fitness, or other per-observation metadata.
            self.observations[slot] = (gene.clone(), *metrics, sequence);
            self.ranks.insert(ArchiveRank::new(value, sequence, slot));
            return Ok(ArchiveOffer::ImprovedDuplicate);
        }

        let worst = if self.observations.len() == self.capacity {
            let worst = *self
                .ranks
                .first()
                .expect("a nonempty full archive has a retained rank");
            if value <= f64::from_bits(worst.score_bits) {
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
        self.ranks.insert(ArchiveRank::new(value, sequence, slot));
        self.next_sequence = next_sequence;
        Ok(outcome)
    }

    fn weekly_observation_preferred(
        objective: &WeeklySearchObjectiveV1,
        incumbent: &WeeklyVectorObservationV1,
        incoming: &WeeklyVectorObservationV1,
    ) -> Result<bool> {
        let ranks = weekly_selection_ranks(
            objective,
            &[incumbent.metrics, incoming.metrics],
            &[incumbent.report.clone(), incoming.report.clone()],
            &[incumbent.gene.fitness, incoming.gene.fitness],
        )?;
        Ok(ranks[1].preference_cmp(&ranks[0]).is_gt())
    }

    /// Merge a whole admitted batch into one bounded union, re-rank exactly
    /// once, and retain. Feasible allzero vectors are never deduplicated here;
    /// unique-vector representatives apply only to infeasible exploration.
    pub(super) fn retain_weekly_batch(
        &mut self,
        objective: &WeeklySearchObjectiveV1,
        batch: Vec<WeeklyVectorObservationV1>,
    ) -> Result<WeeklyArchiveBatchCensusV1> {
        self.weekly_evaluated = self
            .weekly_evaluated
            .checked_add(batch.len())
            .ok_or_else(|| anyhow!("weekly archive evaluated count overflow"))?;
        let mut census = WeeklyArchiveBatchCensusV1 {
            evaluated: batch.len(),
            ..WeeklyArchiveBatchCensusV1::default()
        };
        // Merge by existing behavior identity; improved financial fitness may
        // replace the stored row so the archive never keeps stale measurements.
        let mut by_key: HashMap<EvaluatedGeneBehaviorKey, usize> = self
            .weekly_observations
            .iter()
            .enumerate()
            .map(|(slot, observation)| (EvaluatedGeneBehaviorKey::new(&observation.gene), slot))
            .collect();
        for observation in batch {
            let valid = observation.metrics.iter().all(|value| value.is_finite())
                && observation.gene.fitness.is_finite()
                && observation.report.violation.is_finite()
                && observation.report.trades_per_month.is_finite();
            if !valid {
                census.invalid += 1;
                continue;
            }
            let key = EvaluatedGeneBehaviorKey::new(&observation.gene);
            if let Some(slot) = by_key.get(&key).copied() {
                if Self::weekly_observation_preferred(
                    objective,
                    &self.weekly_observations[slot],
                    &observation,
                )? {
                    self.weekly_observations[slot] = observation;
                }
            } else {
                by_key.insert(key, self.weekly_observations.len());
                self.weekly_observations.push(observation);
            }
        }
        census.distinct = self.weekly_observations.len();
        if self.weekly_observations.is_empty() {
            return Ok(census);
        }
        let metrics: Vec<[f64; 11]> = self
            .weekly_observations
            .iter()
            .map(|observation| observation.metrics)
            .collect();
        let reports: Vec<WeeklyRollingCadenceReportV1> = self
            .weekly_observations
            .iter()
            .map(|observation| observation.report.clone())
            .collect();
        let fitness: Vec<f64> = self
            .weekly_observations
            .iter()
            .map(|observation| observation.gene.fitness)
            .collect();
        let ranks = weekly_selection_ranks(objective, &metrics, &reports, &fitness)?;
        let selected = weekly_select_survivors(&ranks, self.weekly_vector_capacity);
        let mut replace = vec![None; self.weekly_observations.len()];
        for (position, index) in selected.iter().enumerate() {
            replace[*index] = Some(position);
        }
        let mut retained: Vec<Option<WeeklyVectorObservationV1>> =
            (0..selected.len()).map(|_| None).collect();
        let old = std::mem::take(&mut self.weekly_observations);
        for (index, observation) in old.into_iter().enumerate() {
            if let Some(position) = replace[index] {
                retained[position] = Some(observation);
            }
        }
        let retained: Vec<WeeklyVectorObservationV1> = retained
            .into_iter()
            .map(|observation| observation.expect("every selected slot receives one observation"))
            .collect();
        census.capacity_rejected = census.distinct.saturating_sub(retained.len());
        self.weekly_capacity_rejected = self
            .weekly_capacity_rejected
            .checked_add(census.capacity_rejected)
            .ok_or_else(|| anyhow!("weekly capacity-rejected count overflow"))?;
        self.weekly_observations = retained;
        census.retained = self.weekly_observations.len();
        Ok(census)
    }

    pub(super) fn weekly_archive_summary(&self) -> Option<WeeklyArchiveSummaryV1> {
        (self.weekly_vector_capacity > 0).then_some(WeeklyArchiveSummaryV1 {
            capacity: self.weekly_vector_capacity,
            evaluated: self.weekly_evaluated,
            retained: self.weekly_observations.len(),
            capacity_rejected: self.weekly_capacity_rejected,
        })
    }

    pub(super) fn into_finish_observations(self) -> Vec<ArchiveObservation> {
        if self.weekly_vector_capacity > 0 {
            self.weekly_observations
                .into_iter()
                .enumerate()
                .map(|(sequence, observation)| (observation.gene, observation.metrics, sequence))
                .collect()
        } else {
            self.observations
        }
    }

    #[cfg(test)]
    pub(super) fn into_observations(self) -> Vec<ArchiveObservation> {
        self.observations
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn weekly_archive_capacity_retains_feasibility_before_larger_quiet_profit() {
        let mut archive = BoundedProfitableArchive::for_objective(1, true);
        let (mut quiet, q) = observation(0, 1_000.0);
        quiet.fitness = -0.9;
        let (mut active, a) = observation(1, 50.0);
        active.fitness = 1.0;
        assert_eq!(archive.offer(&quiet, &q).unwrap(), ArchiveOffer::Inserted);
        assert_eq!(
            archive.offer(&active, &a).unwrap(),
            ArchiveOffer::ReplacedWorst
        );
        assert_eq!(
            archive.offer(&quiet, &q).unwrap(),
            ArchiveOffer::RejectedCapacity
        );
        assert_eq!(retained_indices(&archive), vec![1]);
    }
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

    fn weekly_vector_gene(index: usize, fitness: f64) -> Gene {
        Gene {
            indices: vec![index],
            weights: vec![0.4],
            long_threshold: 0.25,
            short_threshold: -0.25,
            sl_pips: 20.0,
            tp_pips: 40.0,
            strategy_id: format!("weekly_vector_gene_{index}"),
            fitness,
            ..Gene::default()
        }
    }

    fn weekly_vector_metrics(
        net: f64,
        drawdown: f64,
        daily_drawdown: f64,
        trades: usize,
    ) -> [f64; 11] {
        crate::scoring::weekly_cadence::synthetic_weekly_metrics(
            net,
            drawdown,
            daily_drawdown,
            trades,
        )
    }

    fn weekly_vector_report(windows: u64, trades: usize) -> WeeklyRollingCadenceReportV1 {
        crate::scoring::weekly_cadence::synthetic_weekly_report(windows, trades)
    }

    fn weekly_vector_observation(
        index: usize,
        net: f64,
        drawdown: f64,
        daily_drawdown: f64,
        windows: u64,
        trades: usize,
        fitness: f64,
    ) -> WeeklyVectorObservationV1 {
        WeeklyVectorObservationV1 {
            gene: weekly_vector_gene(index, fitness),
            metrics: weekly_vector_metrics(net, drawdown, daily_drawdown, trades),
            report: weekly_vector_report(windows, trades),
        }
    }

    fn weekly_objective() -> WeeklySearchObjectiveV1 {
        crate::scoring::weekly_cadence::test_weekly_objective()
    }

    #[test]
    fn weekly_vector_archive_flood_retains_one_r_plus_d_and_c() {
        let objective = weekly_objective();
        let mut archive = BoundedProfitableArchive::for_weekly_vector_objective(3);
        let mut batch = vec![
            weekly_vector_observation(0, 50.0, 0.001, 0.001, 2_000, 1, 10.0),
            weekly_vector_observation(1, 100.0, 0.30, 0.02, 1, 445, 20.0),
            weekly_vector_observation(2, -9_900.0, 0.99, 0.049, 0, 5_000, -10.0),
        ];
        for index in 3..13 {
            batch.push(weekly_vector_observation(
                index, 50.0, 0.001, 0.001, 2_000, 1, 10.0,
            ));
        }
        let census = archive.retain_weekly_batch(&objective, batch).unwrap();
        assert_eq!(census.evaluated, 13);
        assert_eq!(census.retained, 3);
        assert_eq!(archive.weekly_vector_len(), 3);
        let observations = archive.weekly_observations();
        assert_eq!(
            observations
                .iter()
                .filter(|observation| observation.report.violating_windows > 1_000)
                .count(),
            1,
            "one unique refuge vector representative"
        );
        assert!(
            observations
                .iter()
                .any(|observation| observation.report.violating_windows == 1)
        );
        assert!(
            observations
                .iter()
                .any(|observation| observation.metrics[0] < -9_000.0)
        );
    }

    #[test]
    fn weekly_vector_archive_union_replaces_arched_weak_cohort_candidate() {
        let objective = weekly_objective();
        let mut archive = BoundedProfitableArchive::for_weekly_vector_objective(2);
        let weak = weekly_vector_observation(0, 50.0, 0.09, 0.02, 2, 445, 10.0);
        let complement = weekly_vector_observation(1, 100.0, 0.02, 0.01, 0, 445, 20.0);
        archive
            .retain_weekly_batch(&objective, vec![weak, complement])
            .unwrap();
        assert_eq!(archive.weekly_vector_len(), 2);
        let strong = weekly_vector_observation(2, 100.0, 0.09, 0.02, 1, 445, 30.0);
        archive
            .retain_weekly_batch(&objective, vec![strong])
            .unwrap();
        let observations = archive.weekly_observations();
        assert_eq!(observations.len(), 2);
        assert!(
            observations
                .iter()
                .any(|observation| observation.gene.strategy_id == "weekly_vector_gene_2"),
            "cross-cohort dominating candidate replaces the weaker archived row"
        );
        assert!(
            !observations
                .iter()
                .any(|observation| observation.gene.strategy_id == "weekly_vector_gene_0")
        );
    }

    #[test]
    fn weekly_vector_archive_improved_behavior_duplicate_replaces_stale_metrics() {
        let objective = weekly_objective();
        let mut archive = BoundedProfitableArchive::for_weekly_vector_objective(1);
        let first = weekly_vector_observation(0, 50.0, 0.09, 0.02, 1, 445, 10.0);
        archive
            .retain_weekly_batch(&objective, vec![first])
            .unwrap();
        let mut improved = weekly_vector_observation(0, 120.0, 0.08, 0.02, 0, 445, 30.0);
        improved.gene.strategy_id = "weekly_vector_gene_0".to_owned();
        archive
            .retain_weekly_batch(&objective, vec![improved])
            .unwrap();
        assert_eq!(archive.weekly_vector_len(), 1);
        assert_eq!(archive.weekly_observations()[0].gene.fitness, 30.0);
        assert_eq!(archive.weekly_observations()[0].report.violating_windows, 0);
    }

    #[test]
    fn weekly_vector_archive_keeps_all_feasible_rows_and_respects_capacity() {
        let objective = weekly_objective();
        let mut archive = BoundedProfitableArchive::for_weekly_vector_objective(3);
        let batch = vec![
            weekly_vector_observation(0, 100.0, 0.09, 0.02, 0, 445, 1.0),
            weekly_vector_observation(1, 100.0, 0.09, 0.02, 0, 445, 3.0),
            weekly_vector_observation(2, 100.0, 0.09, 0.02, 0, 445, 2.0),
        ];
        let census = archive.retain_weekly_batch(&objective, batch).unwrap();
        assert_eq!(census.retained, 3);
        let fitness: Vec<f64> = archive
            .weekly_observations()
            .iter()
            .map(|observation| observation.gene.fitness)
            .collect();
        assert_eq!(fitness, vec![3.0, 2.0, 1.0]);
        assert!(
            archive
                .weekly_observations()
                .iter()
                .all(|observation| observation.report.violating_windows == 0)
        );
    }

    #[test]
    fn weekly_vector_archive_duplicate_prefers_feasible_over_higher_raw_fitness() {
        let objective = weekly_objective();
        let mut archive = BoundedProfitableArchive::for_weekly_vector_objective(1);
        let old_infeasible = weekly_vector_observation(0, 100.0, 0.09, 0.02, 1, 445, 100.0);
        archive
            .retain_weekly_batch(&objective, vec![old_infeasible])
            .unwrap();
        let mut new_feasible = weekly_vector_observation(0, 50.0, 0.09, 0.02, 0, 445, 50.0);
        new_feasible.gene.strategy_id = "weekly_vector_gene_0".to_owned();
        archive
            .retain_weekly_batch(&objective, vec![new_feasible])
            .unwrap();
        assert_eq!(archive.weekly_vector_len(), 1);
        assert_eq!(archive.weekly_observations()[0].report.violating_windows, 0);
        assert_eq!(archive.weekly_observations()[0].gene.fitness, 50.0);
    }

    #[test]
    fn weekly_vector_archive_zero_and_overbatch_capacity_are_reported() {
        let objective = weekly_objective();
        let mut zero = BoundedProfitableArchive::for_weekly_vector_objective(0);
        let batch = vec![weekly_vector_observation(0, 100.0, 0.09, 0.02, 0, 445, 1.0)];
        let census = zero.retain_weekly_batch(&objective, batch).unwrap();
        assert_eq!(census.retained, 0);
        assert_eq!(census.capacity_rejected, 1);
        assert_eq!(zero.weekly_vector_capacity_rejected(), 1);
        assert_eq!(zero.resolved_weekly_vector_capacity(), 0);

        let mut resolved = BoundedProfitableArchive::for_weekly_vector_objective(50_000);
        assert_eq!(
            resolved.resolved_weekly_vector_capacity(),
            WEEKLY_VECTOR_ARCHIVE_CAPACITY_V1
        );
        let batch: Vec<_> = (0..5)
            .map(|index| weekly_vector_observation(index, 100.0, 0.09, 0.02, 1, 445, index as f64))
            .collect();
        let census = resolved.retain_weekly_batch(&objective, batch).unwrap();
        assert_eq!(census.evaluated, 5);
        assert_eq!(census.retained, 5);
        assert_eq!(census.capacity_rejected, 0);
        let mut archive = BoundedProfitableArchive::for_weekly_vector_objective(2);
        let batch: Vec<_> = (0..5)
            .map(|index| weekly_vector_observation(index, 100.0, 0.09, 0.02, 1, 445, index as f64))
            .collect();
        let census = archive.retain_weekly_batch(&objective, batch).unwrap();
        assert_eq!(census.retained, 2);
        assert_eq!(census.capacity_rejected, 3);
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
            assert!(archive.ranks.contains(&ArchiveRank::new(
                archive.retention_value(gene, metrics),
                *sequence,
                slot
            )));
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
