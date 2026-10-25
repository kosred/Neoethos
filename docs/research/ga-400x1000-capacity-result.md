# GA population 400 × 1,000 generations capacity result

Research-only isolated branch; no changes to master, market scoring or final OOS.

GitHub Actions run: https://github.com/kosred/Neoethos/actions/runs/40244417746
CI status: success, 1/1 Rust integration test passed.
Test: crates/neoethos-search/tests/ga_400x1000_capacity_probe.rs

## Measured actual NeoEthos genotype workload

- 400 individuals in each population for 1,000 successive generations.
- 400,000 genome creations/mutations/crossovers using the actual Rust genetic operators.
- 240 feature-slot vocabulary and max 12 weighted indicators per gene.
- 25% randomized immigrants per iteration and 75% crossover/mutation.
- Held full-gene metadata for up to 50,000 archived genomes (the effective archive limit at pop 400 x gen 1000).
- Zero structurally invalid produced genomes.
- Genetics-only execution: 10.030 seconds.
- Genetics-only VmHWM seen inside test: 33,600 kB (about 32.8 MiB).
- Full cargo test process peak maximum RSS: 2,742,440 kB (roughly 2.62 GiB), which includes compilation.
- Full cargo command wall time: 1:08.17, most from compiling/testing setup.
- GitHub-hosted runner reported ~15 GiB available physical RAM at preflight.

## Very important distinction

This proves that the host-side genetic operators can complete 400 x 1000 on the tested runner. It **does not** test historical market evaluations, actual strategy fitness, 400,000 real per-candle backtests, full feature cube, GPU residency, Monte Carlo, internal walk-forward, or broker economics.

The repository's full canonical-research path requires broker-bound canonical acquisition receipts and financial symbol/account snapshots. The publicly released EURUSD Vortex OHLCV lacks that authority. Do not weaken or bypass the broker-truth guard to manufacture a production research result.

The repository developer settings already include 200 population x 1,000 generations, with a 24-hour search budget. Raising population from 200 to 400 roughly doubles raw population evaluation work per generation; memory and wall time depend strongly on M5 rows, features, cost gates, archive, validation folds, and whether GPU is physically available. An actual full 400x1000 search would require its own measured capped pilot with the exact certified data and hardware.

**Result:** GA evolution memory/stability gate PASS. Full historical end-to-end 400x1000 resource gate NOT TESTED. Edge/out-of-sample unchanged.
