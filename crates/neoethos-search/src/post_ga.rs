//! Post-GA admission bounds concurrency, never the number of candidates tested.
//!
//! Full-series signals are regenerated from the frozen gene/input when needed;
//! they are not retained for the whole archive or spilled as huge disk tapes.
use anyhow::{Result, ensure};
use rayon::prelude::*;

// Per full-history row: growing trade vectors (including allocation slack),
// confidence/signal/adaptive arrays, quality's return/drawdown/day aggregates,
// and WF ledger/diagnostic scratch. Shared feature/SMC matrices already resident
// are excluded from *available* headroom, not subtracted a second time here.
// This is a conservative working estimate, NOT an OS reservation/OOM guarantee.
const ROW_WORKING_BYTES: usize = 4 * std::mem::size_of::<crate::quality::Trade>() + 256;
const FIXED_WORKER_BYTES: usize = 1024 * 1024;

fn width_from_headroom(
    rows: usize,
    remaining: usize,
    workers: usize,
    headroom: u64,
) -> Result<usize> {
    if remaining == 0 {
        return Ok(0);
    }
    ensure!(
        rows > 0,
        "post-GA validation requires non-empty historical rows"
    );
    let per_worker = rows
        .checked_mul(ROW_WORKING_BYTES)
        .and_then(|bytes| bytes.checked_add(FIXED_WORKER_BYTES))
        .ok_or_else(|| anyhow::anyhow!("post-GA per-worker allocation estimate overflow"))?;
    // Leave half the measured physical/commit headroom for allocator overhead,
    // other processes and the next shared preparation. Refresh between waves.
    let affordable = usize::try_from((headroom / 2) / per_worker as u64).unwrap_or(usize::MAX);
    ensure!(
        affordable > 0,
        "post-GA validation cannot admit one full-history worker: rows={rows}, estimated_worker_bytes={per_worker}, allocation_headroom_bytes={headroom}; candidate coverage has not been reduced"
    );
    Ok(remaining.min(workers.max(1)).min(affordable))
}

pub(crate) fn post_ga_batch_width(rows: usize, remaining: usize) -> Result<usize> {
    width_from_headroom(
        rows,
        remaining,
        rayon::current_num_threads(),
        neoethos_core::allocation_headroom_bytes(),
    )
}

pub(crate) fn check_cancel() -> Result<()> {
    ensure!(
        !crate::genetic::search_engine::search_cancel_requested(),
        "__DISCOVERY_CANCELLED__ discovery cancelled during post-GA validation"
    );
    Ok(())
}

/// Keep already-unique display IDs unchanged. Generation-time random names can
/// collide, so disambiguate the returned, already-ranked population before any
/// quality/trade/validation artifact is built. The exact genome and candidate
/// index remain unchanged; display IDs are not a substitute for exact identity.
pub(crate) fn disambiguate_candidate_ids(candidates: &mut [(usize, crate::Gene)]) -> usize {
    use std::collections::HashSet;

    let mut reserved: HashSet<String> = candidates
        .iter()
        .map(|(_, gene)| gene.strategy_id.clone())
        .collect();
    let mut observed = HashSet::with_capacity(candidates.len());
    let mut renamed = 0;
    for (position, (candidate_index, gene)) in candidates.iter_mut().enumerate() {
        // An empty identity is malformed, not a name this repair may bless.
        if gene.strategy_id.is_empty() || observed.insert(gene.strategy_id.clone()) {
            continue;
        }
        let original = &gene.strategy_id;
        let mut suffix = 0usize;
        loop {
            let proposed = format!("{original}__candidate_{candidate_index}_{position}_{suffix}");
            if reserved.insert(proposed.clone()) {
                observed.insert(proposed.clone());
                gene.strategy_id = proposed;
                renamed += 1;
                break;
            }
            // Each unsuccessful attempt consumes a distinct reserved name; the
            // finite input bounds attempts without an arbitrary retry ceiling.
            suffix += 1;
        }
    }
    renamed
}

pub(crate) fn trade_equity_curve(
    initial_balance: f64,
    trades: &[crate::quality::Trade],
) -> Vec<f64> {
    if trades.is_empty() {
        return Vec::new();
    }
    let mut equity = initial_balance;
    let mut curve = Vec::with_capacity(trades.len() + 1);
    curve.push(equity);
    for trade in trades {
        equity += trade.pnl;
        curve.push(equity);
    }
    curve
}

/// Parallel waves preserve input order. Broad callers return compact metadata;
/// payload-returning callers must first bound their output set (logs/portfolio).
pub(crate) fn map_bounded<T: Send, R: Send>(
    input: Vec<T>,
    rows: usize,
    map: impl Fn(T) -> Result<R> + Sync,
) -> Result<Vec<R>> {
    let count = input.len();
    let mut input = input.into_iter();
    let mut output = Vec::with_capacity(count);
    while output.len() < count {
        check_cancel()?;
        let width = post_ga_batch_width(rows, count - output.len())?;
        let wave: Vec<_> = input.by_ref().take(width).collect();
        output.extend(wave.into_par_iter().map(&map).collect::<Result<Vec<_>>>()?);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colliding_display_ids_are_unique_without_reordering_or_changing_genomes() {
        let mut candidates = ["same", "same", "same__candidate_8_1_0", "other", "same"]
            .into_iter()
            .enumerate()
            .map(|(position, strategy_id)| {
                (
                    position * 8,
                    crate::Gene {
                        strategy_id: strategy_id.to_owned(),
                        sl_pips: position as f64 + 1.0,
                        ..Default::default()
                    },
                )
            })
            .collect::<Vec<_>>();
        let original = candidates.clone();
        assert_eq!(disambiguate_candidate_ids(&mut candidates), 2);
        let ids = candidates
            .iter()
            .map(|(_, gene)| &gene.strategy_id)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(ids.len(), candidates.len());
        assert_eq!(candidates[0].1.strategy_id, "same");
        assert_eq!(candidates[1].1.strategy_id, "same__candidate_8_1_1");
        assert_eq!(candidates[2].1.strategy_id, original[2].1.strategy_id);
        assert_eq!(candidates[3].1.strategy_id, "other");
        assert_eq!(disambiguate_candidate_ids(&mut candidates), 0);
        for ((actual_index, mut actual), (expected_index, expected)) in
            candidates.into_iter().zip(original)
        {
            assert_eq!(actual_index, expected_index);
            actual.strategy_id = expected.strategy_id.clone();
            assert_eq!(
                serde_json::to_string(&actual).unwrap(),
                serde_json::to_string(&expected).unwrap()
            );
        }
    }

    #[test]
    fn display_id_repair_does_not_turn_missing_identity_into_valid_evidence() {
        let mut candidates = vec![(0, crate::Gene::default()), (1, crate::Gene::default())];
        candidates
            .iter_mut()
            .for_each(|(_, gene)| gene.strategy_id.clear());
        assert_eq!(disambiguate_candidate_ids(&mut candidates), 0);
        assert!(
            candidates
                .iter()
                .all(|(_, gene)| gene.strategy_id.is_empty())
        );
    }

    #[test]
    fn restored_curve_keeps_every_trade_and_empty_is_not_flat_equity() {
        let trades = [50.0, -125.0, 30.0].map(|pnl| crate::quality::Trade {
            pnl,
            ..Default::default()
        });
        assert_eq!(
            trade_equity_curve(100.0, &trades),
            vec![100.0, 150.0, 25.0, 55.0]
        );
        assert!(trade_equity_curve(100.0, &[]).is_empty());
    }

    #[test]
    fn width_limits_concurrency_not_coverage_and_uses_full_tf_rows() {
        let rows = 791_263;
        let cost = (rows * ROW_WORKING_BYTES + FIXED_WORKER_BYTES) as u64;
        assert_eq!(
            width_from_headroom(rows, 50_000, 10, cost * 20).unwrap(),
            10
        );
        assert_eq!(width_from_headroom(rows, 50_000, 10, cost * 6).unwrap(), 3);
        assert_eq!(width_from_headroom(rows, 2, 10, cost * 20).unwrap(), 2);
        assert!(width_from_headroom(rows * 4, 50_000, 10, cost * 20).unwrap() < 10);
    }

    #[test]
    fn unknown_insufficient_and_overflow_headroom_do_not_invent_a_worker() {
        assert!(width_from_headroom(791_263, 1, 10, 0).is_err());
        assert!(width_from_headroom(791_263, 1, 10, 1024).is_err());
        assert!(width_from_headroom(usize::MAX, 1, 10, u64::MAX).is_err());
        assert!(width_from_headroom(0, 1, 10, u64::MAX).is_err());
        assert_eq!(width_from_headroom(0, 0, 10, 0).unwrap(), 0);
    }
}
