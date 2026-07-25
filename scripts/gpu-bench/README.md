# Rented NVIDIA benchmark kit

This directory prepares attributed, fail-fast benchmark runs. It does not claim speedups or select an engine.

1. Run `bash preflight.sh`. By default it requires an RTX A6000-class name, at least 45,000 MiB VRAM, CUDA/Nsight/CUPTI, and executes the real Rust → C ABI → CUDA allocation/upload/kernel/readback smoke tests. The CUDA-gated tests exercise Prototype B, Prototype C and the causal signal/trade traces, reject successful-looking skips, and run Compute Sanitizer memcheck as a fail-loud gate.
2. Create detached, clean historical and candidate worktrees with `bash prepare_worktrees.sh <root> <candidate-sha> [legacy-sha]`.
3. Build the candidate release binary inside its pinned worktree before paid benchmark execution.
4. For real-data fixtures, export one canonical CSV per timeframe with columns `timestamp,high,low,close,<feature...>` and run, for example:
   `python3 prepare_snapshot.py --csv EURUSD_M1.csv --out snapshots/M1.json --timeframe M1 --population 4096`.
   Repeat for H1/M30/M15/M5/M1. The exporter validates ordering and finite values, creates a deterministic candidate population and prints the snapshot SHA-256.
5. Generate the matrix with `python3 run_matrix.py --candidate-sha ... --fixture snapshot --snapshot-dir snapshots`. Prototype-A snapshot jobs are executable. Prototype B/C full-population jobs and the historical legacy adapter remain explicitly blocked until those adapters exist.
6. Use `--execute --skip-blocked` only after inspecting `matrix.json`. Clean timing, diagnostics, Nsight Systems and Nsight Compute remain separate processes/reports.
7. Collate completed JSON reports with `python3 collate.py`.

For fast infrastructure checks, omit `--fixture snapshot`; the deterministic tiny Prototype-A path runs without external data.

The historical reference is pinned to `2be1408ee3986026fdbb2a5a74aaaf6ac67e5209`. Candidate and legacy worktree SHAs are checked before command generation. Missing or unsupported measurements remain blocked or empty; the scripts never fabricate values.
