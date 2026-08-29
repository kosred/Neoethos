use crate::data_selection::CanonicalSearchArtifactScopeV2;
use crate::engine_identity::PopulationEvalEngine;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;
use std::sync::{Arc, Mutex};

pub const POPULATION_ENGINE_RUN_RECEIPT_SCHEMA_VERSION_V1: u16 = 1;

const POPULATION_ENGINE_RECEIPT_HASH_DOMAIN_V1: &[u8] =
    b"neoethos.search.population-engine-run-receipt.v1\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PopulationEngineRunReceiptErrorCodeV1 {
    InvalidCanonicalScope,
    EmptyPopulation,
    OutputCardinalityMismatch,
    NoSuccessfulPopulation,
    RunClosed,
    CountOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PopulationEngineRunReceiptErrorV1 {
    code: PopulationEngineRunReceiptErrorCodeV1,
    message: String,
}

impl PopulationEngineRunReceiptErrorV1 {
    pub const fn code(&self) -> PopulationEngineRunReceiptErrorCodeV1 {
        self.code
    }
}

impl fmt::Display for PopulationEngineRunReceiptErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PopulationEngineRunReceiptErrorV1 {}

fn error(
    code: PopulationEngineRunReceiptErrorCodeV1,
    message: impl Into<String>,
) -> PopulationEngineRunReceiptErrorV1 {
    PopulationEngineRunReceiptErrorV1 {
        code,
        message: message.into(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PopulationEngineRunReceiptV1 {
    schema_version: u16,
    canonical_scope_identity_sha256: String,
    engines: Vec<PopulationEvalEngine>,
    successful_population_count: u64,
    identity_sha256: String,
}

impl PopulationEngineRunReceiptV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn canonical_scope_identity_sha256(&self) -> &str {
        &self.canonical_scope_identity_sha256
    }

    pub fn engines(&self) -> &[PopulationEvalEngine] {
        &self.engines
    }

    pub const fn successful_population_count(&self) -> u64 {
        self.successful_population_count
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }
}

#[derive(Debug)]
struct PopulationEngineRunStateV1 {
    engine_bits: u8,
    successful_population_count: u64,
    closed: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PopulationEngineRunScopeV1 {
    canonical_scope_identity_sha256: Arc<str>,
    state: Arc<Mutex<PopulationEngineRunStateV1>>,
}

fn engine_bit(engine: PopulationEvalEngine) -> u8 {
    match engine {
        PopulationEvalEngine::Cpu => 1 << 0,
        PopulationEvalEngine::CudaNativeF64 => 1 << 1,
        PopulationEvalEngine::CubeclF64 => 1 << 2,
    }
}

fn ordered_engines(bits: u8) -> Vec<PopulationEvalEngine> {
    [
        PopulationEvalEngine::Cpu,
        PopulationEvalEngine::CudaNativeF64,
        PopulationEvalEngine::CubeclF64,
    ]
    .into_iter()
    .filter(|engine| bits & engine_bit(*engine) != 0)
    .collect()
}

fn lock_state(
    scope: &PopulationEngineRunScopeV1,
) -> std::sync::MutexGuard<'_, PopulationEngineRunStateV1> {
    scope
        .state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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

fn receipt_identity(
    canonical_scope_identity_sha256: &str,
    engines: &[PopulationEvalEngine],
    successful_population_count: u64,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(POPULATION_ENGINE_RECEIPT_HASH_DOMAIN_V1);
    hasher.update(POPULATION_ENGINE_RUN_RECEIPT_SCHEMA_VERSION_V1.to_le_bytes());
    hasher.update((canonical_scope_identity_sha256.len() as u64).to_le_bytes());
    hasher.update(canonical_scope_identity_sha256.as_bytes());
    hasher.update((engines.len() as u64).to_le_bytes());
    for engine in engines {
        hasher.update([engine_bit(*engine)]);
    }
    hasher.update(successful_population_count.to_le_bytes());
    hex_lower(&hasher.finalize())
}

pub(crate) fn begin_population_engine_run_v1(
    canonical_scope: &CanonicalSearchArtifactScopeV2,
) -> Result<PopulationEngineRunScopeV1, PopulationEngineRunReceiptErrorV1> {
    canonical_scope.validate().map_err(|source| {
        error(
            PopulationEngineRunReceiptErrorCodeV1::InvalidCanonicalScope,
            format!("population engine run scope is invalid: {source}"),
        )
    })?;
    let identity = canonical_scope.identity_sha256().map_err(|source| {
        error(
            PopulationEngineRunReceiptErrorCodeV1::InvalidCanonicalScope,
            format!("hash population engine run scope: {source}"),
        )
    })?;
    Ok(PopulationEngineRunScopeV1 {
        canonical_scope_identity_sha256: Arc::from(identity),
        state: Arc::new(Mutex::new(PopulationEngineRunStateV1 {
            engine_bits: 0,
            successful_population_count: 0,
            closed: false,
        })),
    })
}

impl PopulationEngineRunScopeV1 {
    pub(crate) fn record_successful_population(
        &self,
        engine: PopulationEvalEngine,
        expected_output_rows: usize,
        actual_output_rows: usize,
    ) -> Result<(), PopulationEngineRunReceiptErrorV1> {
        if expected_output_rows == 0 {
            return Err(error(
                PopulationEngineRunReceiptErrorCodeV1::EmptyPopulation,
                "an empty population is not successful engine evidence",
            ));
        }
        if actual_output_rows != expected_output_rows {
            return Err(error(
                PopulationEngineRunReceiptErrorCodeV1::OutputCardinalityMismatch,
                format!(
                    "population engine produced {actual_output_rows} rows; expected {expected_output_rows}"
                ),
            ));
        }

        let mut state = lock_state(self);
        if state.closed {
            return Err(error(
                PopulationEngineRunReceiptErrorCodeV1::RunClosed,
                "population engine run receipt is already closed",
            ));
        }
        let first_engine_use = state.engine_bits & engine_bit(engine) == 0;
        state.successful_population_count = state
            .successful_population_count
            .checked_add(1)
            .ok_or_else(|| {
                error(
                    PopulationEngineRunReceiptErrorCodeV1::CountOverflow,
                    "successful population count overflow",
                )
            })?;
        state.engine_bits |= engine_bit(engine);
        drop(state);

        if first_engine_use {
            tracing::info!(
                target: "neoethos_search::engine",
                engine = engine.as_str(),
                reproduces_canonical_cpu = engine.reproduces_canonical_cpu(),
                canonical_scope_identity_sha256 = %self.canonical_scope_identity_sha256,
                "population engine recorded after exact successful output"
            );
        }
        Ok(())
    }

    pub(crate) fn finish(
        &self,
    ) -> Result<PopulationEngineRunReceiptV1, PopulationEngineRunReceiptErrorV1> {
        let mut state = lock_state(self);
        if state.closed {
            return Err(error(
                PopulationEngineRunReceiptErrorCodeV1::RunClosed,
                "population engine run receipt is already closed",
            ));
        }
        if state.successful_population_count == 0 {
            return Err(error(
                PopulationEngineRunReceiptErrorCodeV1::NoSuccessfulPopulation,
                "population engine run has no exact successful output",
            ));
        }
        let engines = ordered_engines(state.engine_bits);
        let successful_population_count = state.successful_population_count;
        state.closed = true;
        drop(state);

        let identity_sha256 = receipt_identity(
            &self.canonical_scope_identity_sha256,
            &engines,
            successful_population_count,
        );
        Ok(PopulationEngineRunReceiptV1 {
            schema_version: POPULATION_ENGINE_RUN_RECEIPT_SCHEMA_VERSION_V1,
            canonical_scope_identity_sha256: self.canonical_scope_identity_sha256.to_string(),
            engines,
            successful_population_count,
            identity_sha256,
        })
    }
}
