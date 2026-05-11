# Follow-on Phase 88 - ONNX runtime boundary cleanup

Date: 2026-05-11

## Source gap

The `forex_models_functional` audit flagged that `forex-models/src/lib.rs`
still contained the ONNX inference implementation. That made the crate root do
more than registry/re-export work and kept a legacy runtime boundary inside the
top-level module.

## Changes

- Added a boundary regression test proving the ONNX inference implementation is
  not defined in `lib.rs`.
- Moved `ONNXInferenceEngine` into `forex_models::runtime::onnx`.
- Kept the public API stable with a feature-gated crate-root re-export:
  `forex_models::ONNXInferenceEngine`.
- Added feature-gated `runtime::onnx` module ownership so future ONNX runtime
  work has a dedicated home beside other runtime contracts.

## Verification

- RED: `cargo test -p forex-models runtime::tests::onnx_inference_engine_stays_out_of_crate_root --lib -- --nocapture`
- GREEN: `cargo test -p forex-models runtime::tests::onnx_inference_engine_stays_out_of_crate_root --lib -- --nocapture`
- `cargo check -p forex-models --features onnx`
- `cargo test -p forex-models --lib -- --test-threads=1`
- `cargo fmt --check`
- `git diff --check`
