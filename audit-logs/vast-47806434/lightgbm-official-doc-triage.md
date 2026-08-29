# LightGBM GPU-only failure triage

Date: 2026-08-15

## Official current contract

Context7 resolved the high-reputation official stable LightGBM documentation as
`/websites/lightgbm_readthedocs_io_en_stable`.

- `device_type` accepts three distinct values: `cpu`, `gpu`, and `cuda`.
- `gpu` is the OpenCL implementation; `cuda` is the CUDA implementation.
- CUDA support is a separate Linux build configured with `-DUSE_CUDA=ON` and
  requires an NVIDIA GPU with compute capability 6.0 or later.
- The documented default device is `cpu`; CUDA use therefore must be selected
  explicitly and must not be inferred from the mere presence of a card.
- LightGBM documents GPU accumulation as f32 by default and exposes
  `gpu_use_dp=true` for f64 accumulation at a performance cost. This is a model
  precision boundary that needs its own measured decision; it is not proof of
  the shared NeoEthos f64 feature lane.

Official sources:

- https://lightgbm.readthedocs.io/en/stable/Parameters.html
- https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html

## Current NeoEthos evidence

- `crates/neoethos-models/Cargo.toml` correctly separates `lightgbm3/gpu`
  (OpenCL) from `lightgbm3/cuda` and maps `lightgbm-gpu` to the CUDA feature.
- `LightGBMExpert::effective_device_type()` returns `cuda` only when the
  operator opt-in, CUDA build feature, non-CPU device intent, and visible GPU
  all hold; otherwise it returns `cpu`.
- `fit_internal()` correctly rejects `gpu_only=true` when the resolved device
  is not `cuda`, rather than silently training on CPU.
- `test_gpu_only_mode` supplies only model-local `device=gpu` and
  `gpu_only=true`; it does not install the process-level
  `models.tree_runtime.lightgbm_gpu=true` opt-in and accepts only the older
  no-visible-GPU error text. On the RTX 4090 host the GPU is visible, so the
  earlier visibility guard is bypassed and the later, more precise
  resolved-device-is-CPU diagnostic is returned.

## Candidate result

The compatible dependency candidate fails the isolated test with the precise
resolved-device-is-CPU diagnostic. This is not evidence of a LightGBM training
failure or CPU fallback: training is rejected before the booster call. Exact
baseline attribution is still running and must be compared before deciding
whether any dependency changed this behavior.

## Required regression split

1. A CPU/default-feature contract test must assert that GPU-only intent fails
   closed before training when the CUDA learner or operator opt-in is absent.
2. A real NVIDIA Linux test must build with `--features gpu-cuda`, install the
   resolved NVIDIA assignment and LightGBM opt-in, require
   `device_type=cuda`, train/predict successfully, and record real backend
   evidence with no CPU fallback.
3. `device_type=gpu` must never be used as an alias for CUDA.
4. The `gpu_use_dp` precision/performance choice must be explicit, documented,
   and measured against the CPU/model oracle; it cannot be guessed from memory.
