# Rented NVIDIA benchmark kit

This directory prepares attributed, fail-fast benchmark runs. It does not claim speedups or select an engine.

The paid-run path is **Rust only**. Snapshot preparation, matrix generation, collation and the preflight report are subcommands of `neoethos-cli`. The Python files in this directory (`prepare_snapshot.py`, `run_matrix.py`, `collate.py` and their unit tests) are isolated legacy tooling: they are kept for reference, nothing below depends on them, and no script here invokes them.

1. Run `bash preflight.sh`. By default it requires an RTX A6000-class name, at least 45,000 MiB VRAM, CUDA/Nsight/CUPTI, and executes the real Rust → C ABI → CUDA allocation/upload/kernel/readback smoke tests. The CUDA-gated tests exercise Prototype B, Prototype C and the causal signal/trade traces, reject successful-looking skips, and run Compute Sanitizer memcheck as a fail-loud gate. The report itself is written by `neoethos-cli bench-preflight-report`.
2. Create detached, clean historical and candidate worktrees with `bash prepare_worktrees.sh <root> <candidate-sha> [legacy-sha]`.
3. Build the candidate release binary inside its pinned worktree before paid benchmark execution. Prototype B additionally requires `--features gpu-nvidia`; a binary without it refuses the job rather than measuring something else.
4. For real-data fixtures, export one canonical CSV per timeframe with columns `timestamp,high,low,close,<feature...>` and run, for example:
   `neoethos-cli bench-prepare --csv EURUSD_M1.csv --out snapshots/M1.json --timeframe M1 --population 4096`.
   Repeat for H1/M30/M15/M5/M1. The exporter validates ordering, bounds and finite values, revalidates the result through the same fixture the benchmark consumes, creates a deterministic candidate population and prints the snapshot SHA-256.
5. Generate the matrix with `neoethos-cli bench-matrix --candidate-sha ... --fixture snapshot --snapshot-dir snapshots`. Prototype A, B and C jobs are executable on the candidate; the historical legacy adapter remains explicitly blocked until it exists.
6. Execute the printed commands after inspecting `matrix.json`. Clean timing, diagnostics, Nsight Systems and Nsight Compute remain separate processes and separate reports.
7. Collate completed JSON reports with `neoethos-cli bench-collate --reports cache/gpu-bench/runs --out cache/gpu-bench/summary.json`. Missing fields stay null and parity failures are counted, never averaged away.

`run_rented.sh <candidate-sha> [input-csv-dir]` chains steps 1, 4, 5 and 7 for one session.

For fast infrastructure checks, omit `--fixture snapshot`; the deterministic tiny path runs without external data for all three prototypes.

The historical reference is pinned to `2be1408ee3986026fdbb2a5a74aaaf6ac67e5209`. Candidate and legacy worktree SHAs are checked before command generation. Missing or unsupported measurements remain blocked or empty; the scripts never fabricate values.
