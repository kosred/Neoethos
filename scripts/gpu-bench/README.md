# Rented NVIDIA benchmark kit

This directory prepares attributed, fail-fast benchmark runs. It does not claim speedups or select an engine.

1. Run `./preflight.sh`. By default it requires an RTX A6000-class name, at least 45,000 MiB VRAM, CUDA/Nsight/CUPTI, and executes the real Rust → C ABI → CUDA allocation/upload/kernel/readback smoke test.
2. Create detached, clean historical and candidate worktrees with `./prepare_worktrees.sh <root> <candidate-sha> [legacy-sha]`.
3. Build the candidate release binary inside its pinned worktree before paid benchmark execution.
4. Generate a tiny matrix with `./run_matrix.py --candidate-sha ...`. The manifest marks only Prototype A tiny jobs executable. Historical legacy, B/C, and snapshot jobs remain explicitly blocked until their adapters exist.
5. Use `--execute --skip-blocked` only after inspecting `matrix.json`. Clean timing, diagnostics, Nsight Systems, and Nsight Compute remain separate processes/reports.
6. Collate completed JSON reports with `./collate.py`.

The historical reference is pinned to `2be1408ee3986026fdbb2a5a74aaaf6ac67e5209`. Candidate and legacy worktree SHAs are checked before command generation. Missing or unsupported measurements remain blocked or empty; the scripts never fabricate values.
