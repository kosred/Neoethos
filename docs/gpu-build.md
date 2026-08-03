# Building NeoEthos for full GPU support (discovery GA + ML training)

The code has GPU support for **both** the discovery GA (the cubecl `cubecl_eval`
kernel in `neoethos-search`) **and** ML training (burn deep models + the
gradient boosters). The CPU-only default build **ignores all of it** — you must
build with the right features + the right toolchain present, or everything
silently runs on CPU and never finishes on big data (M1 base = ~5M bars).

This is the canonical recipe. It was hard-won — read the gotchas.

## TL;DR — the build that puts (almost) everything on the GPU

```bash
# Prereqs (one-time, see "Prerequisites" below): NVIDIA driver + CUDA toolkit
# (nvcc) + Boost dev + Vulkan loader. On the 2×A6000 VPS these are present.
cd ~/Neoethos
source $HOME/.cargo/env
export PATH="/usr/local/cuda-12.2/bin:$HOME/.cargo/bin:$PATH"   # nvcc on PATH
export CUDA_HOME=/usr/local/cuda-12.2
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$PWD/target/release/deps:$PWD/target/release:${LD_LIBRARY_PATH:-}"

cargo build --release -p neoethos-cli --features "gpu-vulkan,neoethos-models/gpu-cuda"
```

This combination is deliberate (see "Why this exact feature combo"):
- **`gpu-vulkan`** → search GA kernel on **Vulkan/wgpu** (no libtorch) **and** the
  burn deep models on **Vulkan/wgpu** (`burn-wgpu`).
- **`neoethos-models/gpu-cuda`** → lightgbm / catboost / candle(dqn) / cubecl
  (neat, statistical) on **CUDA**.

### What runs where with this build

**Do not read this table off the page — ask the binary.** It answers for the
features it was actually built with and the hardware actually present:

```bash
neoethos-cli gpu-capabilities
```

Four columns, because there are four different reasons a model lands on the CPU
and each needs a different fix: `declared` (no GPU implementation exists at
all), `compiled` (wrong `--features`), `device` (wrong machine), `opt-in` (the
config value has not been set). The table below is the expected shape.

| On GPU (A6000) | On CPU |
|---|---|
| Discovery GA kernel (Vulkan/cubecl-wgpu) | **lightgbm** — CPU-only *by design*. `lightgbm3/cuda` does not build a GPU tree learner ("[LightGBM] [Fatal] GPU Tree Learner was not enabled in this build") and the OpenCL path does not link. There is no `lightgbm-gpu` feature any more; the one that existed was enabled by nothing and could not have worked. |
| burn deep models: mlp, kan, tabnet, nbeats, nbeatsx_nf, tide, tide_nf, transformer, patchtst, timesnet, exit_agent, sac (Vulkan/burn-wgpu) | sklears_tree, bayes_logit, online_pa, online_hoeffding, isolation_forest, swarm_forecaster, hmm_regime |
| xgboost / xgboost_rf / xgboost_dart **and** meta_blender / probability_calibrator / conformal_gate / meta_stack — all six train through the SAME XGBoost expert, so `xgboost-gpu` (`xgb/cuda`) accelerates all six at once (CUDA) | |
| catboost, catboost_alt (CUDA) | |
| dqn / candle / rlkit (CUDA) — **needs `models.gpu_runtime.rl_device: gpu`** | |
| neat, neuro_evo (cubecl) — **needs `models.gpu_runtime.neuro_evolution_device: gpu`** | |
| elasticnet, logistic (cubecl) — **needs `models.gpu_runtime.statistical_device: gpu`** | |

The three `models.gpu_runtime.*` knobs default to CPU/`auto` on purpose:
compiling a CUDA kernel in must never change what a model produces. The CUDA
paths are linked by `gpu-cuda`; these values decide whether they run. (RL
defaults to `cpu` rather than `auto` because the RL resolver reads `auto` as
"take CUDA device 0", and a CUDA-trained policy is a different policy.)

## Prerequisites

| Component | Why | Check |
|---|---|---|
| NVIDIA driver | the GPUs | `nvidia-smi` |
| **CUDA toolkit (`nvcc`)** matching the runtime (e.g. 12.2) | compiles `lightgbm3/cuda`, `candle-core/cuda`, `cubecl/cuda` | `nvcc --version` — **may already be installed but off PATH** (`/usr/local/cuda-12.2/bin/nvcc`). Add it to PATH, don't reinstall. `apt-get install -y cuda-toolkit-12-2` if truly absent. |
| **Boost dev** (`libboost-dev libboost-filesystem-dev libboost-system-dev`) | LightGBM's GPU/CUDA cmake build requires it | `ls /usr/include/boost/filesystem.hpp` |
| Vulkan loader (`libvulkan.so`) + the GPU visible to Vulkan | `gpu-vulkan` (wgpu) backend for the GA + burn models | `vulkaninfo --summary | grep deviceName` should list the NVIDIA card |

## Why this exact feature combo (the gotchas)

1. **`burn` deep models reach the GPU via `burn-wgpu` (= `gpu-vulkan`).**
   `burn_models.rs` also has a `burn-cuda-backend` alias, and it wins over wgpu
   when both are on — but `gpu-cuda` deliberately does NOT enable it (measured
   2026-06-10 on an A6000: burn-cuda 0.21 re-autotunes every distinct matmul
   shape, 338 autotune passes against 14 real epochs, plus burn-tensor dtype
   panics; the same models take ~17 min total on burn-ndarray). So **`gpu-cuda`
   alone leaves every deep model on CPU** — that is a decision, not an
   oversight, and `gpu-capabilities` reports it as `compiled: false`. Use
   `gpu-vulkan` for the deep models, or `--features gpu-cuda,burn-cuda-backend`
   once the models are big enough to pay for the autotune.

2. **Keep `search` on Vulkan to avoid libtorch.** `neoethos-search/gpu-cuda`
   pulls `dep:tch` (libtorch, for CUDA device enumeration) — a ~2 GB dependency
   that is NOT set up to auto-download (no `download-libtorch` feature). The
   `cli` `gpu-vulkan` feature routes `neoethos-search` to `gpu-vulkan`
   (cubecl-wgpu), so the GA kernel runs on the A6000 via Vulkan **without
   libtorch**. We then add `neoethos-models/gpu-cuda` directly (which does NOT
   pull `tch`) for the CUDA boosters. Hence `--features
   "gpu-vulkan,neoethos-models/gpu-cuda"` rather than the bundled cli `gpu-cuda`.

3. **LightGBM has no GPU path at all.** `neoethos-models/gpu-cuda` originally
   pulled BOTH `lightgbm3/gpu` (OpenCL) and `lightgbm3/cuda`. OpenCL fails to
   LINK (`mold: undefined symbol: clReleaseProgram` — `-lOpenCL` isn't emitted)
   and `lightgbm3/cuda` does not actually produce a GPU tree learner, so
   `effective_device_type()` returns `"cpu"` unconditionally. Both are gone, and
   with them the `lightgbm-gpu` feature that used to key LightGBM's *advertised*
   GPU support on a capability the library does not have.

4. **One feature name per capability.** `gpu-cuda` names
   `reinforcement-learning-cuda`, `neuro-evolution-gpu`, `statistical-gpu` and
   `xgboost-gpu` instead of spelling out `rlkit/cuda` + `candle-core/cuda` +
   `dep:cubecl` + `xgb/cuda` inline. The dependency graph is the same either
   way — but the `#[cfg(feature = …)]` blocks those aggregates guard were
   compiled OUT on every CUDA build, and `registry.rs` (which asks
   `cfg!(feature = "reinforcement-learning-cuda")`) answered "no GPU" for the
   RL, neuro-evolution and statistical families on an RTX 3090. A test
   (`tests/gpu_capability_reachability.rs`) now fails the build when a model
   advertises a capability behind a feature no shipped aggregate can reach.

5. **Runtime env.** At RUN time (not just build) the process needs
   `LD_LIBRARY_PATH` to include `target/release/deps` (the LightGBM `.so`
   sidecar) and `/usr/local/cuda-12.2/lib64`. The `cli` reads
   `enable_gpu_preference: auto` + `tree_device_preference: gpu` from
   `config.yaml`; with a GPU build these route work to the cards (a CPU-only
   build ignores them — that is the trap).

## Simpler alternative: `gpu-vulkan` only (no nvcc, no Boost)

If you don't need the gradient boosters on the GPU (they are fast on a many-core
CPU), this single feature already puts the **heavy** compute on the cards and
needs **no CUDA toolkit / Boost / OpenCL**:

```bash
cargo build --release -p neoethos-cli --features gpu-vulkan
```
→ GA kernel (Vulkan) + all burn deep models (Vulkan) on the A6000. boosters on CPU.

## Verifying it actually used the GPU

`nvidia-smi` **utilization** can read 0% even when the GPU is in use (small
models train in milliseconds between samples). Trust **memory**: a burn model on
the card shows **hundreds of MiB** allocated (vs ~1 MiB idle):
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# 0, 0 %, 313 MiB   ← wgpu/Vulkan context + model on GPU 0 (PROOF)
```
The training log also shows `Burn training: N train, ...` (burn deep model) and,
for the GA, the cubecl client init.

## Known limits / costs

- **GA discovery on GPU OOMs for M1.** The signal/backtest buffer is
  `population × series_rows × 8 B`. On a 46 GB A6000 that fits up to ~M5
  (~800 k rows × pop 4000 ≈ 25 GB); **M1 (~5M rows × 4000 ≈ 160 GB) OOMs** →
  run M1 *discovery* on CPU (or chunk it). M1 *training* is fine (batched).
- **`hmm_regime` is inference-only**, not orchestrator-trainable — do NOT list
  it in `models.ml_models` or `train` hard-fails the whole plan. The trainable
  set is `runtime/capabilities.rs::model_capability` (returns `Some`).
- **Per-combo full-dataset reload.** Each `train`/`discover` invocation reloads
  the symbol's whole dataset (incl. M1's 5M bars) — ~minutes of I/O per combo.
  This dominates wall-clock more than CPU-vs-GPU on the boosters; a future
  optimization is to group a symbol's timeframes into one process.
