//! THE BATCH LEDGER AND THE CANONICAL FEATURE INDEX.
//!
//! Two things a streaming working-set loop cannot be honest without, and
//! neither of them belongs inside `discovery.rs`:
//!
//! 1. **The census.** Every batch the loop touches leaves a named row here —
//!    kept, early-rejected by the predicate, empty, or failed. A batch that is
//!    abandoned without a row is the silent-drop defect one level up, which is
//!    the exact class this project has spent two days closing. The rows are
//!    kept in full (a run is dozens of batches, not millions) so the run-end
//!    census can NAME the abandoned cursors rather than counting them
//!    anonymously.
//!
//! 2. **The canonical feature index** — Option C from
//!    `docs/streaming-parameter-search.md` (settled 2026-08-10). A gene's
//!    `indices: Vec<usize>` are POSITIONS into the ONE
//!    `effective_feature_names` list of the run that produced it. Two batches
//!    build two different lists, so local index 47 denotes a different column
//!    in batch 3 than in batch 11. This type is the run-level list those local
//!    positions are translated into at batch exit.
//!
//! # Why a new ledger rather than `neoethos_data::core::indicator_ledger`
//!
//! `IndicatorLedger` never crosses the crate boundary — the census comment in
//! `discovery::run_discovery_cycle_with_progress` says so in its own words
//! ("only presence is observable" from the search crate). This is a new ledger
//! shaped after it (reason, count, named examples, one census line), not a
//! reuse of that instance.
//!
//! # Relationship to `discovery::BatchRejectionLedger`
//!
//! That ledger is the PREDICATE's own tally: it is written from inside the
//! discovery cycle, where the predicate fires (before the quality screen, the
//! walk-forward and CPCV), and it can only ever see verdicts. This ledger is
//! the LOOP's tally: it also sees the batches whose feature build failed, whose
//! cycle errored, whose portfolio came back empty, and the batches the loop
//! never ran because the machine could not size a working set. Both are logged
//! at run end; neither can account for the other's rows.
//!
//! # The four invariants (doc §"Implement it with these invariants")
//!
//! 1. **Append-only within a run.** `CanonicalFeatureIndex::intern` never
//!    renumbers and never removes: an index, once issued, keeps its meaning for
//!    the life of the run. That is the property whose absence IS the defect.
//! 2. **Remap at BATCH EXIT.** Enforced by where this is called from — the loop
//!    in `orchestration.rs` remaps immediately after a batch's cycle returns and
//!    before the survivor is stored, so no local index is ever held anywhere a
//!    canonical one is expected. A local index that escapes is indistinguishable
//!    from a canonical one and would be silently wrong.
//! 3. **A name that fails to resolve is a HARD ERROR** naming the gene, the
//!    term and the batch — never a skipped term. A gene missing one of its
//!    twelve terms is still structurally valid and would trade a strategy
//!    nobody designed.
//! 4. **Run-end range assertion.** [`CanonicalFeatureIndex::assert_indices_in_range`]
//!    proves every index in every surviving gene addresses a name that exists.

use crate::genetic::Gene;
use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::HashMap;

/// The run-level, append-only feature-name list that survivors from different
/// batches share.
///
/// It grows ONLY as survivors reference names: the final list is exactly the
/// union of names some surviving gene addresses — naturally small, and exactly
/// what `live_portfolio::project_features_to_effective` needs, since the live
/// path projects BY NAME.
#[derive(Debug, Clone, Default)]
pub struct CanonicalFeatureIndex {
    names: Vec<String>,
    by_name: HashMap<String, usize>,
    /// Batches that contributed at least one name. Reported in the census so a
    /// reader can see the list is a union and not one batch's list.
    contributing_batches: usize,
    /// Gene terms translated. The doc's cost claim ("a few hundred hash lookups
    /// per batch") is measurable rather than asserted.
    terms_remapped: usize,
}

impl CanonicalFeatureIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn contributing_batches(&self) -> usize {
        self.contributing_batches
    }

    pub fn terms_remapped(&self) -> usize {
        self.terms_remapped
    }

    pub fn position(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    pub fn into_names(self) -> Vec<String> {
        self.names
    }

    /// INVARIANT 1. Issue (or recall) the canonical index of `name`.
    ///
    /// Append-only: an existing name keeps the index it was first issued, a new
    /// name is appended at the end. Nothing is ever removed, re-sorted or
    /// renumbered, so an index handed out during batch 3 still means the same
    /// column during batch 11.
    fn intern(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.by_name.get(name) {
            return idx;
        }
        let idx = self.names.len();
        self.names.push(name.to_string());
        self.by_name.insert(name.to_string(), idx);
        idx
    }

    /// INVARIANT 2 + 3. Translate one gene's indices from `local_names` (the
    /// batch's own `effective_feature_names`) into this run-level list.
    ///
    /// Returns a CLONE with `indices` rewritten; the input is untouched, so the
    /// batch's own `DiscoveryResult` remains internally consistent and can still
    /// be saved as today's artifact.
    ///
    /// Hard-errors, naming the gene and the term, when:
    ///
    /// * an index addresses past the end of the batch's own name list — the
    ///   only way a term can fail to resolve, and never a reason to drop the
    ///   term and keep the gene;
    /// * a resolved name is empty — an unnamed column cannot be projected by
    ///   name at live time, so admitting it would move the failure from here to
    ///   the trading path;
    /// * `weights` and `indices` disagree in length — a structurally broken
    ///   gene whose remapped copy would silently address a different term set.
    pub fn remap_gene(
        &mut self,
        gene: &Gene,
        local_names: &[String],
        cursor: usize,
    ) -> Result<Gene> {
        if gene.indices.len() != gene.weights.len() {
            bail!(
                "batch at cursor {cursor}: gene '{}' has {} indices but {} weights. Refusing to \
                 remap a structurally inconsistent gene — the remapped copy would address a \
                 different set of terms than the one that was evaluated.",
                gene.strategy_id,
                gene.indices.len(),
                gene.weights.len()
            );
        }
        let mut canonical_indices = Vec::with_capacity(gene.indices.len());
        for (term, &local) in gene.indices.iter().enumerate() {
            let Some(name) = local_names.get(local) else {
                bail!(
                    "batch at cursor {cursor}: gene '{}' term {term} of {} references local \
                     feature index {local}, but that batch's effective_feature_names holds only \
                     {} entries. This term CANNOT be skipped: a gene missing one of its terms is \
                     still structurally valid and would trade a strategy nobody designed.",
                    // `{cursor}`, `{term}` and `{local}` above are INLINE
                    // captures of the bindings in scope; only the three bare
                    // `{}` slots take arguments, in order:
                    //   '{}'          -> strategy_id
                    //   term _ of {}  -> how many terms the gene has
                    //   only {} entries -> how many names the batch has
                    // A fourth `local` was passed here and never consumed
                    // (`error: argument never used`), which is why this file
                    // did not compile.
                    gene.strategy_id,
                    gene.indices.len(),
                    local_names.len()
                );
            };
            if name.trim().is_empty() {
                bail!(
                    "batch at cursor {cursor}: gene '{}' term {term} of {} resolves to an EMPTY \
                     feature name at local index {local}. The live path projects by name \
                     (live_portfolio::project_features_to_effective), so an unnamed column would \
                     fail at trade time instead of here.",
                    // Same shape as the arm above: `{local}` is an inline
                    // capture, so passing `local` again was the `redundant
                    // argument` rustc rejected. Two bare `{}` slots, two args.
                    gene.strategy_id,
                    gene.indices.len()
                );
            }
            canonical_indices.push(self.intern(name));
            self.terms_remapped += 1;
        }
        let mut out = gene.clone();
        out.indices = canonical_indices;
        Ok(out)
    }

    /// Remap a whole batch's surviving portfolio. Called at BATCH EXIT — see
    /// invariant 2. Any single failure fails the batch; nothing partial is
    /// returned, because a partially remapped portfolio is a portfolio with
    /// local indices in it.
    pub fn remap_portfolio(
        &mut self,
        genes: &[Gene],
        local_names: &[String],
        cursor: usize,
    ) -> Result<Vec<Gene>> {
        let before = self.names.len();
        let mut out = Vec::with_capacity(genes.len());
        for gene in genes {
            out.push(self.remap_gene(gene, local_names, cursor)?);
        }
        if self.names.len() > before {
            self.contributing_batches += 1;
        }
        Ok(out)
    }

    /// INVARIANT 4. Every index in every gene addresses a name that exists.
    ///
    /// Cheap, and it closes the class: a run that passes this cannot ship a
    /// portfolio whose genes point past the end of the list the live path will
    /// project with.
    pub fn assert_indices_in_range(&self, genes: &[Gene], stage: &str) -> Result<()> {
        for gene in genes {
            for (term, &idx) in gene.indices.iter().enumerate() {
                if idx >= self.names.len() {
                    bail!(
                        "{stage}: gene '{}' term {term} holds canonical index {idx} but the \
                         run-level feature list has {} names. Either a LOCAL index escaped a \
                         batch without being remapped (invariant 2), or the canonical list was \
                         truncated (invariant 1). Both are fatal — the gene would address a \
                         different column at live time.",
                        gene.strategy_id,
                        self.names.len()
                    );
                }
            }
        }
        Ok(())
    }
}

/// What happened to one batch. Every variant is either a KEPT batch or a NAMED
/// rejection — there is deliberately no "other" bucket, because "other" is how
/// a drop becomes invisible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchOutcome {
    /// The batch produced a portfolio and its survivors were remapped into the
    /// canonical index.
    SurvivorsPromoted,
    /// The early-reject predicate fired inside the discovery cycle (before the
    /// quality screen). The reason and the numbers behind it are in
    /// `discovery::BatchRejectionLedger`; this row records that the LOOP saw it.
    PredicateEarlyReject,
    /// The cycle ran to completion and selected nothing. Not a predicate
    /// rejection: this batch paid for the full screen.
    EmptyPortfolio,
    /// The per-batch feature build failed. Infrastructure, not evidence.
    FeatureBuildFailed,
    /// The discovery cycle returned an error. Infrastructure, not evidence.
    DiscoveryCycleFailed,
    /// The loop stopped before this batch because the caller's batch budget was
    /// spent. Recorded so an interrupted sweep cannot be mistaken for a
    /// completed one.
    BudgetExhausted,
    /// No streaming batch could be sized on this machine (`batch_columns == 0`)
    /// or the parameter space is empty, so the run fell back to the
    /// whole-vocabulary pass. Named rather than silent: the run is still
    /// complete, but it did NOT stream.
    StreamingUnavailable,
}

impl BatchOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SurvivorsPromoted => "survivors_promoted",
            Self::PredicateEarlyReject => "predicate_early_reject",
            Self::EmptyPortfolio => "empty_portfolio",
            Self::FeatureBuildFailed => "feature_build_failed",
            Self::DiscoveryCycleFailed => "discovery_cycle_failed",
            Self::BudgetExhausted => "budget_exhausted",
            Self::StreamingUnavailable => "streaming_unavailable",
        }
    }

    /// True when this batch contributed no survivors — i.e. it was abandoned
    /// and MUST appear by name in the run-end census.
    pub fn is_rejection(self) -> bool {
        !matches!(self, Self::SurvivorsPromoted)
    }

    /// True when the batch actually built a cube and ran a cycle. A run where
    /// this is false for every batch is an infrastructure failure, not an
    /// empty search.
    pub fn reached_the_search(self) -> bool {
        matches!(
            self,
            Self::SurvivorsPromoted
                | Self::PredicateEarlyReject
                | Self::EmptyPortfolio
                | Self::StreamingUnavailable
        )
    }
}

/// Longest error text kept per row. Errors nest and can be paragraphs; the
/// census must stay readable, and the full error is already logged at the
/// point it happened.
const MAX_DETAIL_CHARS: usize = 400;

/// One row per batch. The cursor is the resumable state of the sweep, so a row
/// names WHICH parameter region was abandoned, not merely that one was.
#[derive(Debug, Clone, Serialize)]
pub struct BatchLedgerEntry {
    pub cursor: usize,
    pub next_cursor: usize,
    /// `(indicator, period)` pairs the batch covered.
    pub pairs: usize,
    /// Columns the batch planned to stage, from registry lookups.
    pub planned_columns: usize,
    pub outcome: BatchOutcome,
    /// Genes the batch's portfolio carried (0 for every rejection).
    pub portfolio: usize,
    /// Why, in the words of whatever produced it. Never empty for a rejection.
    pub detail: String,
}

impl BatchLedgerEntry {
    pub fn new(
        cursor: usize,
        next_cursor: usize,
        pairs: usize,
        planned_columns: usize,
        outcome: BatchOutcome,
        portfolio: usize,
        detail: impl Into<String>,
    ) -> Self {
        let mut detail = detail.into();
        if detail.chars().count() > MAX_DETAIL_CHARS {
            detail = detail.chars().take(MAX_DETAIL_CHARS).collect::<String>() + " […truncated]";
        }
        Self {
            cursor,
            next_cursor,
            pairs,
            planned_columns,
            outcome,
            portfolio,
            detail,
        }
    }
}

/// The loop's own census: one row per batch, in the order the loop ran them.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StreamingRunLedger {
    pub entries: Vec<BatchLedgerEntry>,
}

impl StreamingRunLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, entry: BatchLedgerEntry) {
        self.entries.push(entry);
    }

    pub fn batches_seen(&self) -> usize {
        self.entries.len()
    }

    pub fn batches_kept(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome == BatchOutcome::SurvivorsPromoted)
            .count()
    }

    pub fn batches_rejected(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome.is_rejection())
            .count()
    }

    pub fn batches_that_reached_the_search(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome.reached_the_search())
            .count()
    }

    pub fn count_of(&self, outcome: BatchOutcome) -> usize {
        self.entries.iter().filter(|e| e.outcome == outcome).count()
    }

    /// `(reason, count)` for every outcome that occurred, in declaration order.
    pub fn counts_by_outcome(&self) -> Vec<(&'static str, usize)> {
        const ALL: [BatchOutcome; 7] = [
            BatchOutcome::SurvivorsPromoted,
            BatchOutcome::PredicateEarlyReject,
            BatchOutcome::EmptyPortfolio,
            BatchOutcome::FeatureBuildFailed,
            BatchOutcome::DiscoveryCycleFailed,
            BatchOutcome::BudgetExhausted,
            BatchOutcome::StreamingUnavailable,
        ];
        ALL.iter()
            .map(|&o| (o.as_str(), self.count_of(o)))
            .filter(|(_, n)| *n > 0)
            .collect()
    }

    /// Every abandoned batch, by cursor and reason. This is the list that must
    /// exist for a rejection to be a decision rather than a disappearance.
    pub fn rejected_rows(&self) -> Vec<&BatchLedgerEntry> {
        self.entries
            .iter()
            .filter(|e| e.outcome.is_rejection())
            .collect()
    }

    /// The run-end census. Printed whenever ANY batch was seen, including when
    /// none was rejected — "we streamed 40 batches and rejected none" is a
    /// result, and a census that only appears on rejection cannot say it.
    pub fn log_summary(&self, stage: &str) {
        if self.entries.is_empty() {
            return;
        }
        let rejected: Vec<(usize, &'static str, usize, &str)> = self
            .rejected_rows()
            .iter()
            .map(|e| (e.cursor, e.outcome.as_str(), e.pairs, e.detail.as_str()))
            .collect();
        tracing::info!(
            target: "neoethos_search::batch_ledger",
            stage,
            batches_seen = self.batches_seen(),
            batches_kept = self.batches_kept(),
            batches_rejected = self.batches_rejected(),
            counts_by_outcome = ?self.counts_by_outcome(),
            rejected_batches = ?rejected,
            "streaming working-set census — EVERY batch this run touched, by cursor and reason. \
             A cursor that appears here and nowhere else was abandoned deliberately; a cursor \
             that appears nowhere at all is a defect in the loop, not in the search."
        );
    }

    /// Accounting must close before the run's result is believed.
    ///
    /// Two claims, both cheap:
    ///
    /// * every row carries a reason (a rejection with an empty `detail` is an
    ///   anonymous drop wearing a label);
    /// * if batches were seen, at least one reached the search. All of them
    ///   failing to build or to run is an infrastructure failure and must not
    ///   be reported as "the search found nothing" — the same distinction
    ///   `BatchDiscoverySummary::finalize` already draws for the symbol loop.
    pub fn finish(&self, stage: &str) -> Result<()> {
        for entry in &self.entries {
            if entry.outcome.is_rejection() && entry.detail.trim().is_empty() {
                bail!(
                    "{stage}: batch at cursor {} was recorded as '{}' with no reason. Every \
                     rejected batch is counted AND named; a reasonless row is the silent drop \
                     with extra steps.",
                    entry.cursor,
                    entry.outcome.as_str()
                );
            }
        }
        if !self.entries.is_empty() && self.batches_that_reached_the_search() == 0 {
            bail!(
                "{stage}: {} batches were attempted and NONE reached the search \
                 (counts: {:?}). This is an infrastructure failure — a build or a cycle error on \
                 every batch — not an empty search result.",
                self.entries.len(),
                self.counts_by_outcome()
            );
        }
        Ok(())
    }
}

/// One surviving gene, with the batch it came from.
///
/// Provenance is not decoration: "a result that cannot say which parameter
/// region produced it is not a result" (design doc §6).
#[derive(Debug, Clone, Serialize)]
pub struct CanonicalSurvivor {
    /// Cursor of the batch that produced this gene.
    pub source_cursor: usize,
    /// Gene with CANONICAL indices — positions into
    /// `StreamingRunPortfolio::canonical_feature_names`, never into the batch's
    /// own list.
    pub gene: Gene,
}

pub const STREAMING_RUN_PORTFOLIO_SCHEMA_VERSION: u32 = 2;

/// One immutable, batch-local validation snapshot referenced by the run-level
/// streaming artifact. The local gene is authoritative; the remapped gene in
/// [`CanonicalSurvivor`] deliberately is not exact-hash-equivalent.
#[derive(Debug, Clone, Serialize)]
pub struct StreamingBatchValidationSnapshotRefV1 {
    pub source_cursor: usize,
    pub snapshot_root: String,
    pub pointer: crate::validation_snapshot::DiscoveryValidationSnapshotPointerV1,
}

/// Explicit fail-closed boundary for canonical-remapped streaming survivors.
/// A future semantic remap proof may add a new versioned variant; v1 authorizes
/// only the exact local genes stored inside the referenced batch snapshots.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "authority", rename_all = "snake_case")]
pub enum StreamingPromotionAuthorityV1 {
    PerBatchLocalOnly {
        batch_snapshots: Vec<StreamingBatchValidationSnapshotRefV1>,
    },
}

/// The run-level artifact of a streaming sweep: the union portfolio, the
/// canonical name list its genes address, and the full batch census.
#[derive(Debug, Clone, Serialize)]
pub struct StreamingRunPortfolio {
    pub schema_version: u32,
    pub symbol: String,
    pub base_timeframe: String,
    pub higher_timeframes: Vec<String>,
    /// INVARIANT 1's list. Gene indices below are positions into THIS.
    pub canonical_feature_names: Vec<String>,
    pub survivors: Vec<CanonicalSurvivor>,
    pub promotion_authority: StreamingPromotionAuthorityV1,
    /// Cursor the sweep would resume from — one integer, the whole resumable
    /// state.
    pub next_cursor: usize,
    pub space_len: usize,
    pub batch_columns: usize,
    pub ledger: StreamingRunLedger,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gene_with(id: &str, indices: Vec<usize>) -> Gene {
        Gene {
            strategy_id: id.to_string(),
            weights: vec![1.0; indices.len()],
            indices,
            ..Default::default()
        }
    }

    #[test]
    fn intern_is_append_only_and_stable() {
        let mut canon = CanonicalFeatureIndex::new();
        let a = canon.intern("rsi_14");
        let b = canon.intern("atr_20");
        assert_eq!((a, b), (0, 1));
        assert_eq!(canon.intern("rsi_14"), a, "an issued index never changes");
        assert_eq!(canon.intern("atr_20"), b);
        assert_eq!(canon.names(), &["rsi_14".to_string(), "atr_20".to_string()]);
    }

    #[test]
    fn remap_preserves_the_name_every_term_addresses() {
        let mut canon = CanonicalFeatureIndex::new();
        let local: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let gene = gene_with("g1", vec![2, 0]);
        let out = canon.remap_gene(&gene, &local, 0).expect("remap");
        for (term, (&before, &after)) in gene.indices.iter().zip(out.indices.iter()).enumerate() {
            assert_eq!(
                local[before],
                canon.names()[after],
                "term {term} changed which column it addresses"
            );
        }
    }

    #[test]
    fn two_batches_sharing_a_name_share_one_canonical_index() {
        let mut canon = CanonicalFeatureIndex::new();
        let batch3: Vec<String> = ["x", "shared"].iter().map(|s| s.to_string()).collect();
        let batch11: Vec<String> = ["p", "q", "shared"].iter().map(|s| s.to_string()).collect();
        let g3 = canon
            .remap_gene(&gene_with("g3", vec![1]), &batch3, 3)
            .expect("batch 3");
        let g11 = canon
            .remap_gene(&gene_with("g11", vec![2]), &batch11, 11)
            .expect("batch 11");
        assert_eq!(
            g3.indices, g11.indices,
            "the same NAME must resolve to the same canonical index across batches"
        );
        assert_eq!(canon.names().len(), 1, "only referenced names are interned");
    }

    #[test]
    fn an_unresolvable_term_is_a_hard_error_naming_the_gene() {
        let mut canon = CanonicalFeatureIndex::new();
        let local = vec!["only".to_string()];
        let err = canon
            .remap_gene(&gene_with("greedy", vec![0, 7]), &local, 5)
            .expect_err("index 7 cannot resolve against a 1-name list");
        let msg = err.to_string();
        assert!(msg.contains("greedy"), "must name the gene: {msg}");
        assert!(msg.contains("term 1"), "must name the term: {msg}");
        assert!(msg.contains("cursor 5"), "must name the batch: {msg}");
        assert_eq!(
            canon.len(),
            1,
            "the terms interned before the failure stay — the index is append-only, and the \
             caller drops the whole batch"
        );
    }

    #[test]
    fn a_gene_whose_weights_and_indices_disagree_is_rejected() {
        let mut canon = CanonicalFeatureIndex::new();
        let local = vec!["a".to_string(), "b".to_string()];
        let mut gene = gene_with("skew", vec![0, 1]);
        gene.weights.pop();
        let err = canon
            .remap_gene(&gene, &local, 0)
            .expect_err("mismatched weights must not remap");
        assert!(err.to_string().contains("skew"));
    }

    #[test]
    fn range_assertion_catches_an_escaped_local_index() {
        let mut canon = CanonicalFeatureIndex::new();
        let local = vec!["a".to_string()];
        let _ = canon
            .remap_gene(&gene_with("ok", vec![0]), &local, 0)
            .unwrap();
        // A gene that never went through `remap_gene` still carries a LOCAL
        // index — the exact escape invariant 2 exists to prevent.
        let escaped = vec![gene_with("escaped", vec![9])];
        let err = canon
            .assert_indices_in_range(&escaped, "run_end")
            .expect_err("out-of-range index must fail");
        assert!(err.to_string().contains("escaped"), "{err}");
    }

    #[test]
    fn a_reasonless_rejection_fails_the_accounting() {
        let mut ledger = StreamingRunLedger::new();
        ledger.record(BatchLedgerEntry::new(
            0,
            10,
            10,
            100,
            BatchOutcome::PredicateEarlyReject,
            0,
            "   ",
        ));
        let err = ledger.finish("test").expect_err("empty reason must fail");
        assert!(err.to_string().contains("no reason"), "{err}");
    }

    #[test]
    fn every_batch_failing_to_build_is_an_infrastructure_failure() {
        let mut ledger = StreamingRunLedger::new();
        for cursor in [0usize, 10] {
            ledger.record(BatchLedgerEntry::new(
                cursor,
                cursor + 10,
                10,
                100,
                BatchOutcome::FeatureBuildFailed,
                0,
                "disk on fire",
            ));
        }
        let err = ledger
            .finish("test")
            .expect_err("no batch reached the search");
        assert!(err.to_string().contains("infrastructure failure"), "{err}");
    }

    #[test]
    fn a_clean_sweep_closes_its_accounting_and_names_its_rejections() {
        let mut ledger = StreamingRunLedger::new();
        ledger.record(BatchLedgerEntry::new(
            0,
            10,
            10,
            100,
            BatchOutcome::PredicateEarlyReject,
            0,
            "no candidate clears the configured expectancy floor",
        ));
        ledger.record(BatchLedgerEntry::new(
            10,
            20,
            10,
            100,
            BatchOutcome::SurvivorsPromoted,
            3,
            "",
        ));
        ledger.finish("test").expect("accounting closes");
        assert_eq!(ledger.batches_seen(), 2);
        assert_eq!(ledger.batches_kept(), 1);
        assert_eq!(ledger.batches_rejected(), 1);
        assert_eq!(ledger.rejected_rows().len(), 1);
        assert_eq!(ledger.rejected_rows()[0].cursor, 0);
        assert_eq!(
            ledger.counts_by_outcome(),
            vec![("survivors_promoted", 1), ("predicate_early_reject", 1)]
        );
    }
}
