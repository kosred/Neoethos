// ONNX is gone (2026-08-09, batch D4). `runtime/onnx.rs`, the `onnx` Cargo
// feature, the `ort` dependency, `runtime/exports.rs` and `models.export_onnx`
// were removed together: the feature was enabled by no crate, no script and no
// CI job, and the only reachable code — `write_onnx_status_sidecar` — wrote a
// JSON sidecar whose sole content was a note saying no export was attempted.
// Do not re-add a knob here without a consumer that runs.
pub mod artifacts;
pub mod capabilities;
pub mod dispatch;
pub mod hpo;
pub mod prediction;
pub mod profile;
pub mod training_artifact;
