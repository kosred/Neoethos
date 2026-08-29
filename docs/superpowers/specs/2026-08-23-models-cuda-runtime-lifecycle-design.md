# Models CUDA Runtime Lifecycle Design

## Goal

Make every reachable neoethos-models CUDA route select an exact NVIDIA device, execute or fail loudly, and carry the same device truth through training, inference, save, and reload. Explicit CPU remains valid. Auto may choose CPU only when no NVIDIA device is visible.

## Scope

This change owns neoethos-models CUDA policy, model-family routing, persistence, capability reporting, and mandatory device tests. It does not change neoethos-data, neoethos-search, GPU benchmark scripts, or the global training workflow except for the narrow CatBoost construction call that currently overwrites parameters after construction.

The required named CUDA surfaces are:

- XGBoost, XGBoost RF, XGBoost DART
- LightGBM
- CatBoost, CatBoost alternate
- DQN
- NEAT, CR-FM-NES (neuro_evo)
- Logistic, ElasticNet
- Meta blender, probability calibrator, conformal gate, meta stack

## Device policy

CUDA routes use one strict parser. Accepted values are auto, cpu, gpu, gpu:N, cuda, cuda:N, nvidia, and nvidia:N. Empty or malformed ordinals fail. ROCm, Vulkan, Metal, WGPU, unknown strings, negative ordinals, and overflow fail instead of becoming CUDA ordinal zero.

The parser preserves requested intent and yields an optional exact ordinal. The NVIDIA-only visibility probe respects driver visibility masks and never treats ROCm visibility as CUDA availability.

Resolution rules:

- Explicit cpu selects CPU.
- Explicit GPU selects the requested ordinal and every initialization or kernel error is terminal.
- Auto with zero visible NVIDIA devices selects CPU.
- Auto with at least one visible NVIDIA device selects ordinal zero and every initialization or kernel error is terminal.

## Family routing

NEAT, CR-FM-NES, Logistic, and ElasticNet pass the resolved ordinal into CubeCL. DQN passes it into Candle/rlkit. XGBoost uses device=cuda:N both after booster creation and after artifact load. LightGBM uses device_type=cuda and gpu_device_id=N. CatBoost uses task-type GPU and devices=N.

Tree policy parsing no longer maps unknown strings to Auto and no longer discards cuda:N. CatBoost receives its final parameter map at construction, so device preference, GPU-only mode, and ordinal are derived from the same parameters that training uses.

## Persistence

Artifacts retain requested policy, resolved backend/device, and exact ordinal. Reload validates the stored policy and current hardware. If a CUDA route resolves on a present NVIDIA device, reload must initialize and reapply that device. A warning is not a substitute for routing. CPU reload remains valid for Auto artifacts only when no NVIDIA device is visible, subject to GPU-only constraints.

## Capability truth

The registry reports all 15 reachable CUDA surfaces. Supports-GPU means the compiled build contains the route. Prefers-GPU is evaluated separately because LightGBM and statistical models have operator gates; the two concepts must not be forced equal by a test.

## Verification

Source-only RED/GREEN contracts protect strict parsing, Auto resolution, exact ordinals, CatBoost construction, XGBoost reload routing, the 15-name census, and the existence/non-skipping structure of device lifecycle tests.

On RTX 3090, exact gates are split by family. Each gate trains, predicts, saves, reloads, and predicts again where the model supports persistence. It uses explicit GPU or a separately asserted Auto route, cannot skip, asserts a CUDA backend/artifact, and is accompanied by external GPU telemetry.
