# Integrate regime labels into existing discovery pipeline

The repository already has two relevant pieces:

1. `forex-data/src/core/regime_detection.rs`
   - produces `regime_*` feature columns, including volatility state, trend strength, squeeze, mean-reversion/momentum, choppiness, entropy, and regime-change signals.

2. `forex-search/src/discovery.rs`
   - already contains `validate_regime_robustness(trades, features)`.
   - that function is a hard rejection filter: if a strategy loses too much in a regime bucket, it returns false.

For the current architecture this hard rejection is too aggressive. It can kill useful regime-specialist strategies that should remain in the training archive but not necessarily enter live deployment.

## Correct integration direction

Do not add a separate disconnected regime-label pipeline.

Instead:

- keep `validate_regime_robustness` only for deployment/final portfolio approval, if used at all.
- use `genetic::label_strategies_by_regime_windows` after the candidate archive is produced.
- attach or export `StrategyRegimeProfile` beside the existing quality metrics.
- allow `training_candidate = true` even when `deployment_candidate = false`.

## Discovery flow target

Current simplified flow:

```text
search -> ranked candidates -> hard filters -> quality screen -> correlation portfolio
```

Target flow:

```text
search -> large candidate archive -> canonical/diverse archive -> quality screen
       -> regime window labels -> training archive export
       -> strict deployment portfolio selection
```

## Minimal Rust integration sketch

Inside `finalize_candidates_with_progress`, after candidate ranking/truncation and before final portfolio export:

```rust
use crate::genetic::{RegimeLabelPolicy, label_strategies_by_regime_windows, rank_training_profiles};

let eval_cfg = config.evaluation_config(ohlcv.close.last().copied());
let regime_policy = RegimeLabelPolicy::from_env();
let regime_profiles = label_strategies_by_regime_windows(
    features,
    ohlcv,
    &ranked_candidate_genes,
    &eval_cfg,
    regime_policy,
)?;
let regime_profiles = rank_training_profiles(regime_profiles);
```

Then export these profiles to JSON beside the portfolio/quality output, or add them to `DiscoveryResult` once the downstream save format is updated.

## Important behavior rule

A strategy with good 60-180 day windows but weak 8-year aggregate should be:

- kept as `training_candidate` if it has useful specialist windows,
- marked `deployment_candidate = false` unless it passes strict robustness,
- not deleted early by full-period consistency filters.

This is the key distinction between data generation for models and live deployment approval.
