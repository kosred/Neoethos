//! Data-owned, move-only input admission for resident Regime semantic-v3.
//!
//! This validates canonical OHLC and freezes the exact power-of-two scale
//! anchor before CUDA output allocation. It does not compute or materialize a
//! feature value and exposes no context, stream, event or raw device handle.

use super::regime_detection::{
    REGIME_FEATURE_NAMES_V3, REGIME_OPERATION_SCHEDULE_V1, REGIME_SEMANTIC_V3_FIXTURE_SHA256,
    REGIME_SEMANTIC_VERSION, admit_regime_input_v3,
};
use crate::Ohlcv;
use neoethos_gpu_cuda::resident_feature_store_v3::{
    ResidentFeatureColumnBindingV3, ResidentFeatureStoreAssemblerV3,
    ResidentFeatureStoreCudaErrorV3,
};
use neoethos_gpu_cuda::resident_regime_v3::ResidentRegimeRuntimeReceiptV3;
use sha2::{Digest, Sha256};

const RESIDENT_REGIME_INPUT_AUTHORITY_V3: &str =
    "neoethos.data.resident-regime-input-admission.semantic-v3";

/// Crate-owned admission consumed exactly once by resident assembly.
///
/// Private fields and the absence of `Clone`, serde, `Default`, or a public
/// constructor prevent callers from minting or replaying this receipt.
#[must_use = "resident Regime input admission must move into resident assembly"]
#[derive(Debug)]
pub(crate) struct PreparedResidentRegimeInputV3 {
    authority: &'static str,
    row_count: usize,
    scale_anchor: f64,
    input_identity_sha256: [u8; 32],
}

impl PreparedResidentRegimeInputV3 {
    pub(crate) const fn evidence(&self) -> (usize, u64, [u8; 32]) {
        (
            self.row_count,
            self.scale_anchor.to_bits(),
            self.input_identity_sha256,
        )
    }

    pub(crate) fn consume(self) -> (usize, f64, [u8; 32]) {
        debug_assert_eq!(self.authority, RESIDENT_REGIME_INPUT_AUTHORITY_V3);
        (
            self.row_count,
            self.scale_anchor,
            self.input_identity_sha256,
        )
    }

    pub(crate) fn append_to(
        self,
        assembler: &mut ResidentFeatureStoreAssemblerV3,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
    ) -> Result<ResidentRegimeRuntimeReceiptV3, ResidentFeatureStoreCudaErrorV3> {
        let (_row_count, scale_anchor, _input_identity_sha256) = self.consume();
        assembler.append_resident_regime_v3(bindings, scale_anchor)
    }
}

pub(crate) fn preflight_resident_regime_v3(
    ohlcv: &Ohlcv,
) -> anyhow::Result<PreparedResidentRegimeInputV3> {
    let admission = admit_regime_input_v3(ohlcv)?;
    let mut identity = Sha256::new();
    identity.update(RESIDENT_REGIME_INPUT_AUTHORITY_V3.as_bytes());
    identity.update(REGIME_SEMANTIC_VERSION.to_le_bytes());
    identity.update(REGIME_OPERATION_SCHEDULE_V1.as_bytes());
    identity.update(REGIME_SEMANTIC_V3_FIXTURE_SHA256.as_bytes());
    identity.update(admission.row_count().to_le_bytes());
    identity.update(admission.scale_anchor().to_bits().to_le_bytes());
    for name in REGIME_FEATURE_NAMES_V3 {
        identity.update(name.as_bytes());
        identity.update([0]);
    }
    Ok(PreparedResidentRegimeInputV3 {
        authority: RESIDENT_REGIME_INPUT_AUTHORITY_V3,
        row_count: admission.row_count(),
        scale_anchor: admission.scale_anchor(),
        input_identity_sha256: identity.finalize().into(),
    })
}
