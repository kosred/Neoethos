pub mod artifacts;
pub mod capabilities;
pub mod dispatch;
pub mod exports;
pub mod hpo;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod prediction;
pub mod profile;
pub mod training_artifact;

#[cfg(test)]
mod tests {
    #[test]
    fn onnx_inference_engine_stays_out_of_crate_root() {
        let lib_rs = include_str!("../lib.rs");

        assert!(
            !lib_rs.contains("pub struct ONNXInferenceEngine"),
            "ONNX inference implementation belongs in runtime::onnx, not lib.rs"
        );
        assert!(
            lib_rs.contains("pub use runtime::onnx::ONNXInferenceEngine"),
            "lib.rs should only re-export the ONNX inference engine"
        );
    }
}
