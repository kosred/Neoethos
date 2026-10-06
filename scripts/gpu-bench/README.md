# Rented NVIDIA benchmark kit

This directory prepares attributed, fail-fast benchmark runs. It does not claim speedups or select an engine.

The paid-run path is **Rust only**. Snapshot preparation, matrix generation, collation and the preflight report are subcommands of `neoethos-cli`. Superseded Python conversion/matrix/collation implementations were removed after the Rust path replaced them.

1. Run `bash preflight.sh`. By default it accepts an RTX 3090 or RTX A6000 with NVIDIA compute capability at least 8.6 and at least 24,000 reported MiB of physical VRAM, plus CUDA/Nsight/CUPTI. It then runs the count-pinned real-device inventory: f64 native ABI (1), native-B parity (3), CubeCL population parity (7), CubeCL trailing (1), fused (1), direct Prototype A (1), Prototype C (7), the active resident-f64 data suite (192 with 2 explicit ignores), and the HPC indicator sweep (1). Compute Sanitizer separately re-runs native ABI 1/1, native-B 3/3, CubeCL population 7/7, direct Prototype A 1/1, and the full resident Prototype C device group 7/7. Every binary must keep its pinned Cargo count, exit successfully, and report zero memory errors and zero leaked bytes. Any skip, fallback, substitution, zero-test result, count drift, sanitizer error, or leaked byte fails the preflight. Remote runs also capture `nvidia-smi` telemetry; the real-device switches are exported in the test shell before telemetry is backgrounded, so starting telemetry cannot scope those switches away from Cargo. Unknown/lower cards require the explicit `NEOETHOS_ALLOW_OTHER_GPU=1` override. A benchmark that genuinely requires A6000 capacity can narrow the policy with `NEOETHOS_EXPECT_GPU_SUBSTRING='RTX A6000' NEOETHOS_MIN_VRAM_MIB=45000`. The report itself is written by `neoethos-cli bench-preflight-report`.
2. Create detached, clean historical and candidate worktrees with `bash prepare_worktrees.sh <root> <candidate-sha> [legacy-sha]`.
3. Build the candidate release binary inside its pinned worktree before paid benchmark execution. Prototype B additionally requires `--features gpu-nvidia`; a binary without it refuses the job rather than measuring something else.
4. Import each explicitly bar-open source file with the shared `neoethos-cli import` boundary. CSV, TSV, JSON/JSONL, Parquet, Arrow IPC and Vortex source formats may enter there; the importer validates, publishes and reopens an immutable canonical Vortex generation. Then prepare from the exact returned identity, for example:
   `neoethos-cli bench-prepare --data-root cache/import --symbol EURUSD --dataset-identity d1-... --out snapshots/M1.json --timeframe M1 --population 4096`.
   Repeat for each independently sourced direct timeframe. The benchmark command never parses a source format and never manufactures a larger timeframe from M1.
5. Generate the matrix with `neoethos-cli bench-matrix --candidate-sha ... --fixture snapshot --snapshot-dir snapshots` for Prototype B and C only. Prototype A CLI timing is blocked because the current aggregate dispatcher can execute B while attributing the receipt to A; `run_cuda_validation.sh cubecl` uses the direct A engine and is the accepted A hardware proof. The historical legacy adapter remains explicitly blocked until it exists.
6. Execute the printed commands after inspecting `matrix.json`. Clean timing, diagnostics, Nsight Systems and Nsight Compute remain separate processes and separate reports.
7. Collate completed JSON reports with `neoethos-cli bench-collate --reports cache/gpu-bench/runs --out cache/gpu-bench/summary.json`. Missing fields stay null and parity failures are counted, never averaged away.

`run_rented.sh <candidate-sha> [source-dir]` chains steps 1, 4, 5 and 7 for one session. Its directory convention uses one explicitly bar-open CSV per direct timeframe, but those files still pass through the same admitted importer before benchmark preparation.

For fast infrastructure checks, omit `--fixture snapshot`; the deterministic tiny path is available for B and C. Do not add A to a benchmark matrix until its CLI entrypoint directly invokes the A engine.

The historical reference is pinned to `2be1408ee3986026fdbb2a5a74aaaf6ac67e5209`. Candidate and legacy worktree SHAs are checked before command generation. Missing or unsupported measurements remain blocked or empty; the scripts never fabricate values.

## Native CPU / CUDA / HIP correctness fixtures

`native_backend_parity.py` is separate **local correctness tooling**, not a
replacement for the Rust paid benchmark, data importer or production runtime.
It builds once, executes an already-built binary, and compares three pinned
execution receipts. It does not rent hardware or place trades.

| Suite | Actual code under test | Required cases |
| --- | --- | --- |
| `first-hit` | Original CUDA / official HIPIFY `prototype_b.cu` C ABI against independent closed-form expectations and a sequential CPU reference (not the application CPU backtest) | 106 fixtures, 212 ordered results |
| `exact-log` | Existing Rust `quant_log_positive_f64_v3` and shared device `resident_exact_log_v3.cuh`, plus arithmetic edge probes | 48 vectors, including 16 independent accuracy checkpoints |

CUDA builds include `sm_86`, `sm_89`, `sm_120`; this HIP fixture currently targets
`gfx942`, not every AMD card. HIP requires an integrity-checked official
`hipify_sources.py` translation of the current production sources. Translation
diagnostics remain recorded even when translation succeeds.

Examples below run on Linux/WSL from the repository root. Use new absolute output
directories **outside** the repository with an existing parent. Substitute the
actual compiler, manifest, receipt and SHA256 returned by each previous step.

```sh
python3 -B scripts/gpu-bench/native_backend_parity.py build --repo /absolute/repo --suite exact-log --backend cpu --compiler /absolute/rustup/bin/rustc --output /absolute/evidence/cpu-build
python3 -B scripts/gpu-bench/native_backend_parity.py build --repo /absolute/repo --suite exact-log --backend cuda --compiler /absolute/cuda/bin/nvcc --output /absolute/evidence/cuda-build
python3 -B scripts/gpu-bench/native_backend_parity.py build --repo /absolute/repo --suite exact-log --backend hip --compiler /absolute/rocm/bin/hipcc --hip-manifest /absolute/hipify/manifest.json --hip-manifest-sha256 MANIFEST_SHA256 --output /absolute/evidence/hip-build
python3 -B scripts/gpu-bench/native_backend_parity.py run --build-receipt /absolute/evidence/cpu-build/receipt.json --build-sha256 BUILD_SHA256 --output /absolute/evidence/cpu-run
python3 -B scripts/gpu-bench/native_backend_parity.py compare --run-receipt /absolute/evidence/cpu-run/receipt.json --run-sha256 CPU_RUN_SHA256 --run-receipt /absolute/evidence/cuda-run/receipt.json --run-sha256 CUDA_RUN_SHA256 --run-receipt /absolute/evidence/hip-run/receipt.json --run-sha256 HIP_RUN_SHA256
```

Repeat `run` for CUDA and HIP on real compatible devices. For `first-hit`, use
`--suite first-hit`; its CPU compiler is `g++`, not `rustc`. `run` never compiles.
Receipts retain absolute evidence paths: collect them with their full logs or
restore their recorded directory layout before comparing on another host.

Exit zero from `build` means **compiled and linked only**. Missing device/driver,
failed API, missing cases, altered sources/binary/logs, or a numerical/discrete
mismatch must fail. Cross-backend outputs are bit-exact; the mathematical log
accuracy allowance of one ULP is a separate comparison to real-log checkpoints,
not permission for CPU/CUDA/HIP disagreement. Strict no-contraction and denormal
flags follow [NVIDIA floating-point guidance](https://docs.nvidia.com/cuda/floating-point/index.html)
and [AMD Clang floating-point controls](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.2.3/LLVM/clang/html/UsersManual.html#controlling-floating-point-behavior).

Neither suite proves complete indicator/SMC, genetic search, model training,
walk-forward/OOS, risk or per-trade ledger parity. The Rust HIP application owner
and lifecycle integration remain separate requirements. Unit tests in
`test_native_backend_parity.py` use explicitly synthetic records to reject false
passes; those tests are **not GPU execution evidence**.
