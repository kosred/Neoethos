# Model Phase 1 Status

Timestamp: 2026-04-08 18:19:18 +02:00

## Done

- Tree models hardening completed and verified:
  - `lightgbm`
  - `xgboost`
  - `catboost`
- Shared runtime confidence / abstain gate is now used across:
  - tree runtime predictions
  - adaptive runtime predictions
  - DQN runtime predictions
  - linear statistical runtime predictions
  - bayesian statistical runtime predictions
  - exit-agent runtime predictions
- `swarm_forecaster` was upgraded with:
  - learned validation weights
  - stricter stale-artifact validation
  - better artifact rebuild behavior
- `isolation_forest` was upgraded with a robust score profile:
  - mean
  - std
  - median
  - MAD
- `online_hoeffding` runtime reporting now reflects the effective fallback blend state truthfully.
- `dqn` fallback backend naming now reflects the actual fallback basis truthfully.

## Verified

- `cargo test -p forex-models -- --nocapture`
- `cargo clippy -p forex-models --all-targets -- -D warnings`

Current result:

- `forex-models` test suite: green
- `forex-models` clippy: green

## Remaining In Phase 1

- Reduce remaining degraded fallback dependence in:
  - `streaming/adaptive_impl.rs`
  - `rl/dqn_impl.rs`
  - `forecasting/swarm_impl.rs`
- Revisit tree models later if we want a true pure-Rust native-equivalent engine instead of surrogate fallback when external native backends are unavailable.
- Improve GPU execution reality:
  - strategy search is still primarily CPU-first in the normal discovery path
  - GPU search path exists, but is not yet the dominant/default runtime path
  - model GPU acceleration is still uneven across families

## Next Intended Order

1. Finish residual Phase 1 model simplifications.
2. Clean repo junk / dead files / obsolete Python leftovers.
3. Decompose oversized config surface.
4. Re-check subsystems against latest upstream docs.
