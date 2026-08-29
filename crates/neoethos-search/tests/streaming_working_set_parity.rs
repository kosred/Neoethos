//! THE STREAMING WORKING-SET LOOP: PARITY FIRST, THEN THE SWEEP.
//!
//! # What this file pins
//!
//! 1. **PARITY.** With streaming off, the loop performs exactly ONE feature
//!    build — with NO working set installed — and exactly ONE discovery cycle,
//!    and hands back the cycle's own result untouched. That is today's path,
//!    expressed as an assertion instead of as a claim. Everything else in this
//!    file is only worth reading if this passes.
//!
//! 2. **The canonical index (Option C).** A gene's `indices` are positions into
//!    the feature-name list of the batch that produced it. Across batches those
//!    positions mean different columns. The loop remaps every survivor at BATCH
//!    EXIT onto one append-only run-level list, and the property that must hold
//!    is not "the index is preserved" — it is **"the NAME the term addresses is
//!    preserved"**. That is what these tests assert.
//!
//! 3. **The anti-no-op guard.** A streaming loop whose feature build ignores
//!    the installed working set is worse than no loop: every batch computes the
//!    same columns, the cursor marches through a parameter space nothing ever
//!    touched, and the census reports a sweep that never happened. The failure
//!    is silent by construction, so the loop checks for it and hard-errors.
//!
//! # Why the fake result type
//!
//! The loop is generic over [`BatchSearchResult`] precisely so these properties
//! can be tested without a real GA run: the remap, the census and the guard are
//! the things under test, and a real discovery cycle would test the GA instead
//! while adding minutes of runtime and a data dependency.

use neoethos_data::FeatureFrame;
use neoethos_search::Gene;
use neoethos_search::orchestration::{
    BatchOutcome, BatchSearchResult, CanonicalFeatureIndex, StreamingPlan,
    run_streaming_working_set,
};

struct FakeResult {
    names: Vec<String>,
    portfolio: Vec<Gene>,
}

impl BatchSearchResult for FakeResult {
    fn local_feature_names(&self) -> &[String] {
        &self.names
    }

    fn portfolio_genes(&self) -> &[Gene] {
        &self.portfolio
    }
}

fn gene(id: &str, indices: Vec<usize>) -> Gene {
    Gene {
        strategy_id: id.to_string(),
        weights: vec![1.0; indices.len()],
        indices,
        ..Default::default()
    }
}

fn frame(names: &[String]) -> FeatureFrame {
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
        neoethos_data::test_fixtures::canonical_test_timestamps(2),
        names.to_vec(),
        ndarray::Array2::<f64>::zeros((2, names.len())),
    )
    .expect("valid f64 test frame")
}

/// Rows the batch width is sized against. Only used to ask the hardware what it
/// can afford — no frame of this size is ever built here.
const BUDGET_ROWS: usize = 200_000;

/// Can this machine stream at all? `streaming_batch_columns` is free RAM
/// divided by the row width, so on a small box it is legitimately 0 and the
/// loop falls back to one whole-vocabulary pass. The tests below branch on this
/// rather than assuming, so they assert something true on every machine.
fn can_stream() -> bool {
    neoethos_data::core::hpc_ta::streaming_batch_columns(BUDGET_ROWS) > 0
        && neoethos_data::core::hpc_ta::extended_sweep_space_len() > 0
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. PARITY
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn parity_streaming_off_is_one_build_one_cycle_and_no_working_set() {
    let mut builds = 0usize;
    let mut with_working_set = 0usize;
    let mut cycles = 0usize;
    let local: Vec<String> = ["alpha", "beta", "gamma"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let outcome = run_streaming_working_set(
        &StreamingPlan::disabled(),
        BUDGET_ROWS,
        |batch| {
            builds += 1;
            if batch.is_some() {
                with_working_set += 1;
            }
            Ok(frame(&local))
        },
        |features| {
            cycles += 1;
            assert_eq!(features.names.len(), 3);
            Ok(FakeResult {
                names: local.clone(),
                portfolio: vec![gene("winner", vec![2, 0])],
            })
        },
    )
    .expect("the single-pass path must not fail");

    assert_eq!(builds, 1, "exactly one feature build, as today");
    assert_eq!(
        with_working_set, 0,
        "no working set may be installed when streaming is off — that is what makes the \
         column layout identical to today's"
    );
    assert_eq!(cycles, 1, "exactly one discovery cycle, as today");
    assert!(!outcome.streamed);
    assert_eq!(outcome.batches.len(), 1);
    assert_eq!(outcome.ledger.batches_seen(), 1);
    assert_eq!(outcome.ledger.batches_rejected(), 0);

    // The result handed back is the cycle's own, untouched: its genes still
    // carry LOCAL indices and still address its own name list.
    let result = &outcome.batches[0].result;
    assert_eq!(result.portfolio[0].indices, vec![2, 0]);
    assert_eq!(result.names, local);

    // And the canonical copy addresses the same NAMES.
    let canonical_gene = &outcome.batches[0].canonical_portfolio[0];
    let names = outcome.canonical.names();
    assert_eq!(names[canonical_gene.indices[0]], "gamma");
    assert_eq!(names[canonical_gene.indices[1]], "alpha");
}

#[test]
fn parity_an_empty_portfolio_still_reaches_the_caller_and_is_named() {
    let local = vec!["alpha".to_string()];
    let outcome = run_streaming_working_set(
        &StreamingPlan::disabled(),
        BUDGET_ROWS,
        |_batch| Ok(frame(&local)),
        |_features| {
            Ok(FakeResult {
                names: local.clone(),
                portfolio: Vec::new(),
            })
        },
    )
    .expect("an empty portfolio is a result, not an error");

    assert_eq!(
        outcome.batches.len(),
        1,
        "the caller's own empty-portfolio guard must be the one that fires"
    );
    assert_eq!(outcome.ledger.count_of(BatchOutcome::EmptyPortfolio), 1);
    assert_eq!(outcome.ledger.rejected_rows().len(), 1);
    assert!(
        !outcome.ledger.rejected_rows()[0].detail.trim().is_empty(),
        "a rejection with no reason is an anonymous drop"
    );
}

#[test]
fn parity_a_build_error_propagates_exactly_as_today() {
    let err = run_streaming_working_set::<FakeResult, _, _>(
        &StreamingPlan::disabled(),
        BUDGET_ROWS,
        |_batch| anyhow::bail!("vortex file missing"),
        |_features| unreachable!("the cycle must not run after a failed build"),
    )
    .expect_err("the single pass must propagate, not swallow");
    assert!(err.to_string().contains("vortex file missing"), "{err}");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. THE CANONICAL INDEX — Option C's four invariants
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_same_name_in_two_batches_resolves_to_one_canonical_index() {
    let mut canonical = CanonicalFeatureIndex::new();
    // Batch 3 and batch 11 hold the same column at DIFFERENT local positions —
    // the whole defect in one line.
    let batch3: Vec<String> = ["ema_20", "rsi_14"].iter().map(|s| s.to_string()).collect();
    let batch11: Vec<String> = ["atr_10", "cci_30", "rsi_14"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let from3 = canonical
        .remap_portfolio(&[gene("g3", vec![1])], &batch3, 3)
        .expect("batch 3 remap");
    let from11 = canonical
        .remap_portfolio(&[gene("g11", vec![2])], &batch11, 11)
        .expect("batch 11 remap");

    assert_eq!(
        from3[0].indices, from11[0].indices,
        "local 1 in batch 3 and local 2 in batch 11 are the SAME column — after the remap they \
         must be the same canonical index"
    );
    assert_eq!(
        canonical.names(),
        &["rsi_14".to_string()],
        "the canonical list is the union of names survivors actually reference, not the union \
         of every batch's vocabulary"
    );
    canonical
        .assert_indices_in_range(&from3, "test")
        .expect("invariant 4");
    canonical
        .assert_indices_in_range(&from11, "test")
        .expect("invariant 4");
}

#[test]
fn an_index_once_issued_never_changes_meaning() {
    let mut canonical = CanonicalFeatureIndex::new();
    let first: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
    let second: Vec<String> = ["c", "b", "a"].iter().map(|s| s.to_string()).collect();

    let g1 = canonical
        .remap_portfolio(&[gene("g1", vec![0, 1])], &first, 0)
        .expect("first");
    let names_after_first: Vec<String> = canonical.names().to_vec();
    let g2 = canonical
        .remap_portfolio(&[gene("g2", vec![0, 2])], &second, 7)
        .expect("second");

    assert_eq!(
        &canonical.names()[..names_after_first.len()],
        names_after_first.as_slice(),
        "append-only: earlier entries keep their positions"
    );
    // g1 term 0 was "a"; g2 term 1 is "a" — same canonical index, still "a".
    assert_eq!(canonical.names()[g1[0].indices[0]], "a");
    assert_eq!(canonical.names()[g2[0].indices[1]], "a");
    assert_eq!(g1[0].indices[0], g2[0].indices[1]);
}

#[test]
fn a_term_that_cannot_be_resolved_fails_loudly_instead_of_being_skipped() {
    let mut canonical = CanonicalFeatureIndex::new();
    let local = vec!["only_one".to_string()];
    let err = canonical
        .remap_portfolio(&[gene("twelve_termer", vec![0, 4])], &local, 9)
        .expect_err("a term addressing past the end of the batch's list must fail");
    let msg = err.to_string();
    assert!(msg.contains("twelve_termer"), "must name the gene: {msg}");
    assert!(msg.contains("cursor 9"), "must name the batch: {msg}");
    assert!(
        msg.contains("CANNOT be skipped"),
        "must say why dropping the term is not an option: {msg}"
    );
}

#[test]
fn run_end_range_assertion_catches_a_local_index_that_escaped() {
    let mut canonical = CanonicalFeatureIndex::new();
    let local = vec!["a".to_string()];
    canonical
        .remap_portfolio(&[gene("ok", vec![0])], &local, 0)
        .expect("remap");
    let escaped = vec![gene("escapee", vec![41])];
    let err = canonical
        .assert_indices_in_range(&escaped, "run end")
        .expect_err("an unremapped local index must be caught");
    assert!(err.to_string().contains("escapee"), "{err}");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. THE SWEEP ITSELF
// ─────────────────────────────────────────────────────────────────────────────

/// Names a batch's own columns would be emitted under, so a fake build can
/// stand in for the real one without pretending the working set does nothing.
fn names_for(batch: &neoethos_data::core::hpc_ta::SweepBatch) -> Vec<String> {
    batch
        .pairs
        .iter()
        .map(|pair| format!("{}_{}", pair.id, pair.period))
        .collect()
}

#[test]
fn the_sweep_advances_and_carries_survivors_across_batches() {
    if !can_stream() {
        // Not a skip: on a machine that cannot size a batch the loop MUST fall
        // back to one whole-vocabulary pass and SAY SO. That is the assertion
        // here.
        let local = vec!["a".to_string()];
        let outcome = run_streaming_working_set(
            &StreamingPlan::streaming(4),
            BUDGET_ROWS,
            |_batch| Ok(frame(&local)),
            |_features| {
                Ok(FakeResult {
                    names: local.clone(),
                    portfolio: vec![gene("g", vec![0])],
                })
            },
        )
        .expect("fallback must succeed");
        assert!(!outcome.streamed);
        assert_eq!(
            outcome.ledger.count_of(BatchOutcome::StreamingUnavailable),
            1,
            "the fallback must be NAMED, never silent"
        );
        return;
    }

    let mut cursors_built: Vec<usize> = Vec::new();
    let mut batch_names: Vec<Vec<String>> = Vec::new();
    let outcome = run_streaming_working_set(
        &StreamingPlan::streaming(2),
        BUDGET_ROWS,
        |batch| {
            let batch = batch.expect("the streaming path always installs a working set");
            cursors_built.push(batch.cursor);
            let names = names_for(&batch);
            batch_names.push(names.clone());
            Ok(frame(&names))
        },
        |features| {
            let names: Vec<String> = features.names.clone();
            // One gene addressing the batch's FIRST column — a different real
            // column in every batch, which is exactly the case that used to be
            // unrepresentable in one portfolio.
            Ok(FakeResult {
                portfolio: vec![gene("survivor", vec![0])],
                names,
            })
        },
    )
    .expect("the sweep must not fail");

    assert!(outcome.streamed);
    assert!(!cursors_built.is_empty(), "at least one batch must run");
    assert_eq!(outcome.batches.len(), cursors_built.len());
    assert_eq!(outcome.ledger.batches_kept(), cursors_built.len());
    if cursors_built.len() > 1 {
        assert!(
            cursors_built[1] > cursors_built[0],
            "the cursor must ADVANCE — a repeating batch re-explores a region the run already \
             searched and reports it as new"
        );
        assert_ne!(
            batch_names[0], batch_names[1],
            "two batches must not compute the same columns"
        );
    }

    // Every survivor addresses, by canonical index, the name its own batch gave
    // it — across batches, in one list.
    let canonical_names = outcome.canonical.names();
    for (batch, names) in outcome.batches.iter().zip(batch_names.iter()) {
        let local_index = batch.result.portfolio[0].indices[0];
        let canonical_index = batch.canonical_portfolio[0].indices[0];
        assert_eq!(
            canonical_names[canonical_index], names[local_index],
            "the remap changed which column a term addresses"
        );
    }
    assert_eq!(outcome.survivor_count(), outcome.batches.len());
    for survivor in outcome.survivors() {
        assert!(
            cursors_built.contains(&survivor.source_cursor),
            "every survivor must name the batch that produced it"
        );
    }
}

#[test]
fn a_build_that_ignores_the_working_set_is_a_hard_error() {
    if !can_stream() {
        return;
    }
    // A frame that contains none of the batch's columns: what the run would see
    // if the sweep still took the budget-capped prefix and ignored the
    // installed batch.
    let unrelated = vec!["close".to_string(), "hour of day".to_string()];
    let err = run_streaming_working_set::<FakeResult, _, _>(
        &StreamingPlan::streaming(1),
        BUDGET_ROWS,
        |_batch| Ok(frame(&unrelated)),
        |_features| unreachable!("the cycle must never run on an unobserved working set"),
    )
    .expect_err("an ignored working set must be fatal, not tolerated");
    let msg = err.to_string();
    assert!(
        msg.contains("working set was not observed"),
        "the error must say what is wrong: {msg}"
    );
    assert!(
        msg.contains("cursor"),
        "and which batch it happened on: {msg}"
    );
}
