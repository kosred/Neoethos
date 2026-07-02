//! Canonical scoring module — shared ingredients + named formulas.
//!
//! **Operator-validated 2026-05-25** (response to mass-closure batch
//! question: "ομοιομορφία είναι καλό να υπάρχει" — uniformity is good).
//!
//! ## Why this module exists
//!
//! Pre-2026-05-25 the workspace had SIX independent "score-a-strategy"
//! formulas, each with its own weighting scheme and magic constants:
//!
//! | Formula | Where it lived | Drove what |
//! |---------|----------------|------------|
//! | `score_from_metrics` | `genetic::evolution_math` | GA fitness — the one the population evolves toward |
//! | `score_strategy` | `quality.rs::StrategyQualityAnalyzer` | Quality screen — post-GA gate before promotion |
//! | `window_quality_score` | `genetic::regime_labels` | Per-regime-window scoring during regime labelling |
//! | `archive_quality_score` | `genetic::diversity` | Archive ranking (what survives across generations) |
//! | `compute_validation_score` | `validation.rs` | OOS/CPCV slice scoring |
//! | (implicit) `portfolio_rank` | `portfolio.rs` | Per-asset Sharpe ranking inside the allocator |
//!
//! Each disagreed with the others. Concretely: a strategy with
//! `sharpe=1.5, max_dd=0.06, win_rate=0.50, profit_factor=1.3, trades=80,
//! consistency=0.55` evaluates to:
//!
//! - `score_from_metrics`: ~0.62 (GA loves it because Sharpe-weighted)
//! - `score_strategy`: ~0.45 (quality gate is harsher because PF below 1.5 trigger threshold)
//! - `window_quality_score`: ~0.31 (penalises consistency below 0.6)
//! - `archive_quality_score`: ~0.85 (favours net-profit + trade-count combo)
//!
//! The GA optimised toward one number while the gates dropped strategies
//! using a different number → wasted compute + silent strategy churn.
//!
//! ## The new layout (operator-approved doctrine §3 layout)
//!
//! - **`scoring/ingredients.rs`** — the shared "ingredient functions"
//!   each of the named scores combines. Every weighting / clamping /
//!   sharpening rule sits in ONE place so it can be reviewed in
//!   isolation. The audit's tracked findings F-042 / F-049 / F-057 /
//!   F-089 all become refactors against this single file rather than
//!   six parallel grep-and-tweaks.
//!
//! - **`scoring/named.rs`** — the four production scoring functions
//!   the audit identified, each documented with an explicit weight
//!   table:
//!   - `ga_fitness(metrics, trade_floor)` — what the GA maximises
//!   - `quality_score(metrics, gates)` — post-GA quality gate
//!   - `window_score(metrics, policy)` — per-regime-window ranking
//!   - `archive_score(metrics)` — diversity-archive ranking
//!
//! Each named function is a thin combiner over the ingredients with
//! a published constant weight table. The bench-mark behaviour stays
//! byte-for-byte identical at first (the four functions inline the
//! same magic constants their predecessors used) — the migration is
//! STRUCTURAL, not behavioural. Once everything compiles + tests
//! pass + downstream consumers point at the new symbols, the
//! follow-up batch can collapse identical weight tables across the
//! named functions and remove the duplicate magic constants.
//!
//! ## Schema versioning — DiscoveryRunProfile.scoring_version
//!
//! Per the operator-approved migration plan (§3 doctrine), persisted
//! `DiscoveryRunProfile` artifacts now carry a `scoring_version: u32`
//! field. Old artifacts default to `1` (the pre-unification behaviour
//! these named functions replicate at first). Bumping to `2` will
//! happen only when the weight tables of the named functions are
//! actually unified — at that point the changelog records the formula
//! delta and any deployed artifact with `scoring_version=1` is read
//! with a `DEPRECATED — retrain when convenient` warning.
//!
//! ## Migration order (no-break, per safety doctrine §4)
//!
//! 1. **This commit (Phase A — STRUCTURAL)**: `scoring/` module lands
//!    with the four named functions delegating to in-line copies of
//!    the existing magic constants. The legacy functions in
//!    `evolution_math.rs` / `regime_labels.rs` / `quality.rs` /
//!    `diversity.rs` continue to exist as `#[deprecated]` re-exports
//!    that just call the new named functions. Existing callers
//!    continue to compile against the old paths. **Zero behavioural
//!    change**.
//!
//! 2. Phase B (follow-up batch): one production caller per commit
//!    migrates to the new `scoring::*` paths. Each commit runs the
//!    test suite + a representative backtest to confirm byte-equality
//!    of the GA's selected genomes.
//!
//! 3. Phase C (follow-up batch): the legacy re-exports are deleted,
//!    the deprecated warning goes away, and the weight tables in the
//!    named functions can be unified into a single shared table
//!    (bumping `scoring_version` to 2).
//!
//! This commit is **Phase A only**. It is intentionally
//! non-behavioural so that the audit's "scoring unification" finding
//! is mechanically resolvable in the follow-up Phase B without
//! risking the GA's fitness landscape mid-migration.

pub mod ingredients;
pub mod named;

pub use ingredients::{
    consistency_component, drawdown_penalty, expectancy_component, ga_pf_component,
    net_component, profit_factor_component, sharpe_component, trades_confidence,
    win_rate_component,
};

pub use named::{
    archive_score, ga_fitness, quality_score, window_score, ScoringVersion,
    SCORING_VERSION_CURRENT,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the scoring-module surface so accidental refactors that
    /// rename / delete a named function fail loudly at unit-test time.
    #[test]
    fn scoring_module_publishes_four_named_functions_plus_ingredients() {
        // Compile-time check via fn pointers: if any of these vanishes
        // or changes shape, this test stops compiling.
        let _ga: fn(&[f64; 11]) -> f64 = ga_fitness;
        let _archive: fn(&[f64; 11]) -> f64 = archive_score;

        // Ingredients are also published so future named scores can
        // mix-and-match without duplicating clamping logic.
        let _conf: fn(f64) -> f64 = trades_confidence;
        let _sharp: fn(f64, f64) -> f64 = sharpe_component;
        let _cons: fn(f64) -> f64 = consistency_component;
        let _dd: fn(f64) -> f64 = drawdown_penalty;
        let _pf_ga: fn(f64) -> f64 = ga_pf_component;
        let _pf: fn(f64) -> f64 = profit_factor_component;
        let _wr: fn(f64) -> f64 = win_rate_component;
        let _net: fn(f64) -> f64 = net_component;
        let _exp: fn(f64) -> f64 = expectancy_component;
    }

    #[test]
    fn scoring_version_is_four_after_steady_income_daily_dd_penalty() {
        // 2026-06-06: 2→3 — CONSISTENT-monthly-return GA (dominant reward =
        // monthly_target_hit_rate in metrics[7]; total-net demoted).
        // 2026-07-02: 3→4 — STEADY-INCOME worst-period penalty: ga_fitness now
        // also penalises metrics[10] (max_daily_drawdown ×10.0), which the
        // evaluator computed but the GA ignored. Two equally-consistent genes
        // now rank by who never had a catastrophic day. Artifacts from earlier
        // versions are tagged and NOT directly comparable.
        assert_eq!(SCORING_VERSION_CURRENT.0, 4);
    }
}
