# Evo/search/GPU fix queue

This branch is for the next correctness pass in the strategy-search pipeline that feeds downstream models.

## Priority 1: genetic offspring hygiene

Target: `crates/forex-search/src/genetic/evolution_math.rs`

Changes to apply:

1. Add a helper that resets all derived metrics on a newly created child:
   - fitness
   - sharpe_ratio
   - win_rate
   - max_drawdown
   - profit_factor
   - expectancy
   - trades_count
   - slice_pass_rate
   - consistency

2. Call this helper in `crossover` instead of resetting only `fitness`.

3. Call this helper in `mutate` before returning the mutated gene.

4. Call `mutated.normalize(n_indicators, 1)` before returning from `mutate`.

5. Keep `new_random_gene` normalized before it leaves the constructor path.

Reason: child genes should never carry stale parent scores into selection, archiving, or downstream training data.

## Priority 2: archive dedup by canonical signature

Target: `crates/forex-search/src/genetic/search_engine.rs`

Changes to apply:

1. Import `gene_signature_hash`.
2. Replace `seen_strategy_ids: HashSet<String>` with `seen_archive_signatures: HashSet<u64>`.
3. Before archive insertion, clone/normalize the gene or rely on already-normalized gene, then compute `gene_signature_hash`.
4. Skip archive insertion if the signature already exists.

Reason: `strategy_id` is intentionally randomized, so duplicate/canonical-equivalent strategies can still enter the archive if dedup is based only on id.

## Priority 3: CUDA causal timing parity

Target: `crates/forex-search/src/cubecl_eval.rs`

Issue to verify:

The CPU evaluator uses the prior bar signal (`signals[i - 1]`) before filling on the current bar. The CUDA full backtest kernel should use the same timing. If it still reads `signals_flat[signal_base + i]`, it should be changed to read the prior signal row when opening a new position.

Reason: CPU and CUDA backtest paths must rank the same strategy using the same execution timing, otherwise GPU-selected candidates can be biased before feeding models.

## Priority 4: downstream model feed contract

Before exporting strategies to model training, each candidate should carry both:

- selection score (`fitness`)
- real financial metrics (`net_profit`, drawdown, PF, trades, expectancy, consistency)

Reason: models should not learn from a field named `fitness` as if it were net profit when it is actually a rank/selection score.
