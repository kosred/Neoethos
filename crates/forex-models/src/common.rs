//! Shared helpers for forex-models internals.
//!
//! Phase 67 extraction: `flatten_features` was duplicated across
//! `evolution::crfmnes_gpu`, `evolution::neat_gpu`, and
//! `statistical::linear_gpu`. Each emitted a different error label
//! ("neuro-evo cuda…", "NEAT cuda…", "statistical cuda…") but the
//! validation + flattening math was identical.

use anyhow::{Result, bail};
use ndarray::Array2;

/// Validate that `features` has exactly `input_dim` columns and return
/// the flattened row-major buffer ready for upload to a CUDA kernel.
///
/// `caller_label` is folded into the error message so the operator
/// knows which subsystem produced the mismatch (e.g. `"NEAT"` for the
/// neuro-evolution path, `"statistical"` for linear softmax).
pub fn cuda_flatten_features(
    features: &Array2<f32>,
    input_dim: usize,
    caller_label: &str,
) -> Result<Vec<f32>> {
    if features.ncols() != input_dim {
        bail!(
            "{caller_label} cuda feature dimension mismatch: expected {}, received {}",
            input_dim,
            features.ncols()
        );
    }
    Ok(features.iter().copied().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn flatten_accepts_matching_dimension() {
        let features = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let flat = cuda_flatten_features(&features, 3, "test").expect("matching dim");
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn flatten_rejects_mismatched_dimension() {
        let features = array![[1.0_f32, 2.0, 3.0]];
        let err = cuda_flatten_features(&features, 5, "neuro-evo")
            .expect_err("mismatched dim must reject");
        let msg = err.to_string();
        assert!(msg.contains("neuro-evo"));
        assert!(msg.contains("expected 5"));
        assert!(msg.contains("received 3"));
    }
}
