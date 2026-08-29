use crate::engine_identity::PopulationEvalEngine;
use crate::native_population_residency_receipt_v1::NativePopulationResidencyReceiptV1;
use crate::population_engine_run_receipt_v1::PopulationEngineRunReceiptV1;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;

pub const EXACT_POPULATION_EXECUTION_RUN_RECEIPT_SCHEMA_VERSION_V2: u16 = 2;

const EXECUTION_RUN_RECEIPT_HASH_DOMAIN_V2: &[u8] =
    b"neoethos.search.exact-population-execution-run-receipt.v2\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExactPopulationExecutionRunReceiptErrorCodeV2 {
    MissingNativeResidencyReceipt,
    UnexpectedNativeResidencyReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExactPopulationExecutionRunReceiptErrorV2 {
    code: ExactPopulationExecutionRunReceiptErrorCodeV2,
    message: String,
}

impl fmt::Display for ExactPopulationExecutionRunReceiptErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ExactPopulationExecutionRunReceiptErrorV2 {}

/// Durable V2 carrier. The nested V1 receipt is unchanged; native residency is
/// additive and cannot be defaulted, reconstructed, or upcast from V1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExactPopulationExecutionRunReceiptV2 {
    schema_version: u16,
    engine_receipt_v1: PopulationEngineRunReceiptV1,
    native_residency_receipt_v1: Option<NativePopulationResidencyReceiptV1>,
    identity_sha256: String,
}

impl ExactPopulationExecutionRunReceiptV2 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn engine_receipt_v1(&self) -> &PopulationEngineRunReceiptV1 {
        &self.engine_receipt_v1
    }

    pub const fn native_residency_receipt_v1(&self) -> Option<&NativePopulationResidencyReceiptV1> {
        self.native_residency_receipt_v1.as_ref()
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub fn engines(&self) -> &[PopulationEvalEngine] {
        self.engine_receipt_v1.engines()
    }

    pub fn canonical_scope_identity_sha256(&self) -> &str {
        self.engine_receipt_v1.canonical_scope_identity_sha256()
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

pub(crate) fn seal_exact_population_execution_run_receipt_v2(
    engine_receipt_v1: PopulationEngineRunReceiptV1,
    native_residency_receipt_v1: Option<NativePopulationResidencyReceiptV1>,
) -> Result<ExactPopulationExecutionRunReceiptV2, ExactPopulationExecutionRunReceiptErrorV2> {
    let used_cuda = engine_receipt_v1
        .engines()
        .contains(&PopulationEvalEngine::CudaNativeF64);
    match (used_cuda, native_residency_receipt_v1.is_some()) {
        (true, false) => {
            return Err(ExactPopulationExecutionRunReceiptErrorV2 {
                code: ExactPopulationExecutionRunReceiptErrorCodeV2::MissingNativeResidencyReceipt,
                message: "CudaNativeF64 engine evidence has no sealed native residency receipt"
                    .to_owned(),
            });
        }
        (false, true) => {
            return Err(ExactPopulationExecutionRunReceiptErrorV2 {
                code:
                    ExactPopulationExecutionRunReceiptErrorCodeV2::UnexpectedNativeResidencyReceipt,
                message:
                    "CPU/CubeCL-only engine evidence cannot carry a native CUDA residency receipt"
                        .to_owned(),
            });
        }
        (true, true) | (false, false) => {}
    }

    let mut hasher = Sha256::new();
    hasher.update(EXECUTION_RUN_RECEIPT_HASH_DOMAIN_V2);
    hasher.update(EXACT_POPULATION_EXECUTION_RUN_RECEIPT_SCHEMA_VERSION_V2.to_le_bytes());
    hasher.update((engine_receipt_v1.identity_sha256().len() as u64).to_le_bytes());
    hasher.update(engine_receipt_v1.identity_sha256().as_bytes());
    match native_residency_receipt_v1.as_ref() {
        None => hasher.update([0]),
        Some(receipt) => {
            hasher.update([1]);
            hasher.update((receipt.identity_sha256().len() as u64).to_le_bytes());
            hasher.update(receipt.identity_sha256().as_bytes());
        }
    }
    let identity_sha256 = hex_lower(&hasher.finalize());
    Ok(ExactPopulationExecutionRunReceiptV2 {
        schema_version: EXACT_POPULATION_EXECUTION_RUN_RECEIPT_SCHEMA_VERSION_V2,
        engine_receipt_v1,
        native_residency_receipt_v1,
        identity_sha256,
    })
}
