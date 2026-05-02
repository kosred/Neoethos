# CPU/GPU strategy search optimization notes

This branch starts the search optimization work by removing mathematically redundant genetic candidates before hashing and evaluation.

## Immediate change

`Gene::normalize` now canonicalizes the linear part of each gene:

- indicator indices are clamped into range
- indicator terms are sorted
- duplicate indicator indices are merged by summing their weights
- invalid or non-finite weights are repaired
- invalid thresholds and stop/target values are repaired

This reduces wasted CPU search without removing genuinely different strategies. Equivalent rules now hash the same way after normalization, so `SeenSignatureMemory` becomes more effective.

## Review status

The current branch is ahead of `master` and contains only the canonical gene normalization patch plus this optimization note. The normalization direction is correct and safe for search efficiency because it removes equivalent linear representations instead of removing real strategy variants.

Known follow-up items still open:

- `crossover` currently resets only `fitness`; derived metrics should also be reset before the child is evaluated again.
- `mutate` should normalize the child before it is returned, because indicator replacement can create duplicate or invalid terms.
- `profitable_archive` should deduplicate by `gene_signature_hash` after normalization, not by `strategy_id`.
- Some code paths still use `fitness` as both selection score and profit proxy; this should be split before trusting filters as financial filters.
- Backtesting still needs an explicit execution lag so signals produced on bar `i` cannot enter on the same close of bar `i`.

## Next CPU-side steps

1. Normalize every crossover/mutation child before archive insertion.
2. Deduplicate archives by `gene_signature_hash`, not `strategy_id`.
3. Separate actual money metrics from selection scores.
4. Add explicit `execution_lag_bars` to backtesting.
5. Move feature prefiltering inside train folds only.
6. Bound novelty/diversity work so large searches do not become O(n^2).
7. Use staged Monte Carlo validation: small screening first, large perturbation count only for survivors.

## Safe CPU profile for long local runs

For a one-month CPU search, prefer broad exploration with strict duplicate control instead of over-selecting the first lucky candidates:

```bash
FOREX_BOT_PROP_SEEN_RETRY=64
FOREX_BOT_PROP_ARCHIVE_CAP=50000
FOREX_BOT_PROP_RANDOM_IMMIGRANTS=0.30
FOREX_BOT_PROP_SURVIVOR_FRACTION=0.08
FOREX_BOT_PROP_PARENT_SELECTION=rank
FOREX_BOT_PROP_SURVIVOR_SELECTION=rank
FOREX_BOT_PROP_SELECTION_TEMPERATURE=0.75
FOREX_BOT_NOVELTY_WEIGHT=0.0
```

For persistent duplicate memory across runs:

```bash
FOREX_BOT_PROP_SEEN_FILE=.local/search_seen_signatures.bin
FOREX_BOT_PROP_SEEN_LOAD_MAX=20000000
FOREX_BOT_PROP_SEEN_MAX_ENTRIES=20000000
```

## GPU parity target

The fast GPU path should become a stage-1 candidate generator, not final proof. GPU candidates should be converted into the same canonical strategy representation and then validated through the same SMC, SL/TP, spread, commission, timestamp, walk-forward and stress-test logic as CPU/Gene strategies.

## Runtime reporting target

Every search run should report requested backend, actual backend, GPU feature availability, CUDA device count, whether GPU was used, fallback reason, signal backend, backtest backend and final validation backend.
