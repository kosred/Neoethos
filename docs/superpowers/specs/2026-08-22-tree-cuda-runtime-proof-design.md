# Tree CUDA Runtime Proof Design

## Goal

Prove that the production XGBoost and LightGBM experts train and predict on an
NVIDIA device in `gpu-cuda` builds, and fail loudly when that exact device path
cannot be used.

## Chosen design

Use one dedicated integration-test binary. It installs one immutable model
runtime configuration with `device = "cuda"`, `gpu_only = true`, an exact
visible-device count, and LightGBM CUDA permission. Each expert trains a small
typed `FeatureFrame`, predicts three-class probabilities, persists its normal
runtime artifact, and the test verifies that the artifact records the CUDA
device decision. Any probe, training, prediction, or artifact mismatch fails
the test; there is no skip or CPU fallback.

XGBoost uses its current `tree_method = hist` plus `device = cuda`. Its runtime
probe must exercise that same production spelling. A failed CUDA probe becomes
a typed training error in GPU-only mode instead of silently resolving CPU.
LightGBM continues to use `device_type = cuda` and its existing pre-allocation
GPU-only check.

## Alternatives rejected

- Compile-only verification cannot prove a native library actually selected
  or launched its CUDA learner.
- Allowing the existing XGBoost CPU fallback would let a rented GPU run pass
  without exercising GPU training.
- Separate ad-hoc binaries would duplicate fixtures and device policy without
  strengthening the proof.

## Verification

First observe a focused RED on the current XGBoost fallback/legacy probe.
Then make the smallest production change and run the dedicated test on the RTX
3090 with warning denial and 100 ms GPU telemetry. Inspect all output in
INFO, WARNING, ERROR order and require both persisted artifacts to name CUDA,
valid probability matrices, nonzero device activity, and no skip/fallback text.

