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
| On GPU (A6000) | On CPU |
|---|---|
| Discovery GA kernel (Vulkan/cubecl-wgpu) | xgboost / xgboost_rf / xgboost_dart (the `xgb` crate has no GPU build wired here) |
| burn deep models: mlp, kan, tabnet, nbeats, nbeatsx_nf, tide, tide_nf, transformer, patchtst, timesnet (Vulkan/burn-wgpu) — on a CUDA-only build these stay on CPU **until the device policy says `gpu`**, see gotcha 1 | sklears_tree |
| lightgbm, catboost, catboost_alt (CUDA) | a few custom CPU-only: online_pa/hoeffding, meta_blender/stack, probability_calibrator, conformal_gate |
| dqn / candle / rlkit (CUDA) | |
| neat, statistical (cubecl) | |

## Prerequisites

| Component | Why | Check |
|---|---|---|
| NVIDIA driver | the GPUs | `nvidia-smi` |
| **CUDA toolkit (`nvcc`)** matching the runtime (e.g. 12.2) | compiles `lightgbm3/cuda`, `candle-core/cuda`, `cubecl/cuda` | `nvcc --version` — **may already be installed but off PATH** (`/usr/local/cuda-12.2/bin/nvcc`). Add it to PATH, don't reinstall. `apt-get install -y cuda-toolkit-12-2` if truly absent. |
| **Boost dev** (`libboost-dev libboost-filesystem-dev libboost-system-dev`) | LightGBM's GPU/CUDA cmake build requires it | `ls /usr/include/boost/filesystem.hpp` |
| Vulkan loader (`libvulkan.so`) + the GPU visible to Vulkan | `gpu-vulkan` (wgpu) backend for the GA + burn models | `vulkaninfo --summary | grep deviceName` should list the NVIDIA card |

## Why this exact feature combo (the gotchas)

1. **`burn` deep models now reach CUDA too — but only when ASKED.**
   *(Changed 2026-08-02 with the burn 0.22 bump. What this item used to say —
   "there is no wired burn-CUDA backend in this repo, so `gpu-cuda` alone
   leaves every deep model on CPU" — was true until then and is why the TL;DR
   recipe pairs `gpu-vulkan` with `neoethos-models/gpu-cuda`.)*

   `gpu-cuda` now enables `burn-cuda-backend`, so a CUDA build has a
   `Device::cuda()` to offer. Two things follow, and both matter:

   - **The device is chosen at RUNTIME, not by the build.** In 0.21 the
     backend was a TYPE, so a `gpu-cuda` binary claimed CUDA whether or not a
     card answered. In 0.22 `resolve_device_policy` asks the machine
     (`Device::enumerate`) and falls back to the ndarray CPU floor, recording
     `("cpu", "ndarray_cpu")` honestly, when nothing answers.
   - **The default is still CPU.** The shipped device policy is `auto`, and
     `auto` is deliberately gated out of the CUDA lane
     (`enumerate_accelerator` in `burn_models.rs`). To put burn training on
     the card set the per-model `device` param — equivalently
     `with_device_policy` — to `gpu`, `cuda` or `gpu:N`.

   **Re-measure before you flip it.** The one measurement on record (A6000,
   2026-06-10) found burn-cuda pathological for these tiny nets: 338 autotune
   passes against 14 real epochs in 74 minutes on one combo, versus ~17
   minutes TOTAL on CPU. That is burn 0.21 / cubecl 0.10 and has NOT been
   repeated on 0.22 / cubecl 0.11. Also watch peak VRAM across the whole run
   (`nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1`, take the
   maximum): `cubecl.toml` no longer pins `max_streams = 1`, and burn #4991 —
   the real fix for the multi-pool VRAM leak — landed two days after the
   0.22.0-pre.1 cut, so it is not in this version.

   `gpu-vulkan` remains the path for non-NVIDIA cards, and still drives the
   burn models via `burn-wgpu`.

2. **Keep `search` on Vulkan to avoid libtorch.** `neoethos-search/gpu-cuda`
   pulls `dep:tch` (libtorch, for CUDA device enumeration) — a ~2 GB dependency
   that is NOT set up to auto-download (no `download-libtorch` feature). The
   `cli` `gpu-vulkan` feature routes `neoethos-search` to `gpu-vulkan`
   (cubecl-wgpu), so the GA kernel runs on the A6000 via Vulkan **without
   libtorch**. We then add `neoethos-models/gpu-cuda` directly (which does NOT
   pull `tch`) for the CUDA boosters. Hence `--features
   "gpu-vulkan,neoethos-models/gpu-cuda"` rather than the bundled cli `gpu-cuda`.

3. **Drop lightgbm's OpenCL path — keep CUDA.** `neoethos-models/gpu-cuda`
   originally pulled BOTH `lightgbm3/gpu` (OpenCL) and `lightgbm3/cuda`. The
   OpenCL path fails to LINK (`mold: undefined symbol: clReleaseProgram`) because
   `-lOpenCL` isn't emitted. We want CUDA anyway, so `lightgbm3/gpu` is removed
   from the `gpu-cuda` feature in `crates/neoethos-models/Cargo.toml` (keep
   `lightgbm3/cuda`). If you re-add OpenCL, you must also link `-lOpenCL`.

4. **Runtime env.** At RUN time (not just build) the process needs
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
