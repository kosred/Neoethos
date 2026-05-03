# CPU/GPU strategy search optimization notes

This note records the current optimization state after merging the existing strategy-search CPU/GPU fixes into `master`.

## Current merged baseline

`master` now includes the `claude/fix-strategy-search-cpu-gpu-rtkhl` branch with CPU/GPU parity, causal preprocessing, deterministic seed handling, segment fixes, and parallelized GPU discovery/refinement work.

Important merged fixes include:

- `GpuDiscoveryConfig::seed` for deterministic CPU/GPU search reproducibility.
- Causal `shift_down` + causal z-score preprocessing before GPU discovery consumes feature frames.
- Segment off-by-one fix so the recent window reaches the last available bar.
- Parallelized expensive population/refinement loops with Rayon.
- GPU feature build type-check repair using the proper `FeatureProfile` path.
- Non-finite metric hardening before GA sorting.

## Remaining follow-up from Ariadne review

The older `ariadne/search-cpu-gpu-fast-fixes` branch still contains a useful canonical `Gene::normalize` patch, but it conflicts with the newer `strategy_gene.rs` from the merged Claude branch. It should be re-applied carefully rather than force-merged.

Open items still worth addressing before trusting long expensive runs:

1. Canonicalize gene indicator terms: sort indicator indices, merge duplicate indicator indices, clamp invalid weights, and repair invalid thresholds/SL/TP.
2. Ensure crossover and mutation reset all derived metrics, not only fitness.
3. Normalize mutated children before hashing, archiving, or evaluation.
4. Deduplicate profitable archives by `gene_signature_hash` after normalization, not by `strategy_id`.
5. Keep `fitness` as a selection score and financial net profit as a separate metric/field.
6. Make sure final validation uses the same SMC-gated signal path as search, not a simplified `signals_for_gene` path.
7. Keep GPU discovery as a fast stage-1 candidate generator and route survivors through the full CPU/Rust validator.

## Runtime target

The practical target is not maximum theoretical search size. The target is a pipeline that can produce a candidate shortlist in 24-72 hours, then validate that shortlist on locked forward data and live/paper forward testing.

Recommended search shape:

- Stage 1: very broad GPU/CPU candidate generation.
- Stage 2: strict duplicate removal and cheap filters.
- Stage 3: full evaluator with spread, commission, SL/TP, timestamps, SMC gates and realistic execution timing.
- Stage 4: walk-forward/CPCV/purge/embargo validation.
- Stage 5: locked forward block and paper forward watchlist.

## Suggested local run profile

```bash
FOREX_BOT_PROP_SEEN_RETRY=64
FOREX_BOT_PROP_ARCHIVE_CAP=50000
FOREX_BOT_PROP_RANDOM_IMMIGRANTS=0.30
FOREX_BOT_PROP_SURVIVOR_FRACTION=0.08
FOREX_BOT_PROP_PARENT_SELECTION=rank
FOREX_BOT_PROP_SURVIVOR_SELECTION=rank
FOREX_BOT_PROP_SELECTION_TEMPERATURE=0.75
FOREX_BOT_NOVELTY_WEIGHT=0.0
FOREX_BOT_PROP_SEEN_FILE=.local/search_seen_signatures.bin
FOREX_BOT_PROP_SEEN_LOAD_MAX=20000000
FOREX_BOT_PROP_SEEN_MAX_ENTRIES=20000000
```
