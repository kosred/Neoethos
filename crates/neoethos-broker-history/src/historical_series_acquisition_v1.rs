//! Immutable acquisition authority for direct cTrader canonical trendbars.
//!
//! This module contains no connector, credential, tick, quote, resampling, or
//! mutable-current lookup. It binds exact broker generations already published
//! by the one-cell service into a resumable plan and a final matrix authority.

use crate::bootstrap_writer::CTraderTrendbarProvenanceV1;
use anyhow::{Context, Result, bail};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity,
    CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe, SelectedDatasetGenerationV1,
    open_exact_dataset_generation,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const CANONICAL_TRENDBAR_SERIES_FROM_MS_V1: i64 = 1_451_606_400_000;
pub const CANONICAL_TRENDBAR_PAGING_POLICY_V1: &str =
    "ctrader.trendbars.oldest-exclusive.has-more.v1";

const PLAN_SCHEMA_V1: &str = "neoethos.canonical-trendbar-acquisition-plan.v1";
const CHECKPOINT_SCHEMA_V1: &str = "neoethos.canonical-trendbar-acquisition-checkpoint.v1";
const MATRIX_SCHEMA_V1: &str = "neoethos.canonical-trendbar-matrix.v1";
const PLAN_RECEIPT_SCHEMA_V1: &str = "neoethos.canonical-trendbar-plan-receipt.v1";
const CHECKPOINT_RECEIPT_SCHEMA_V1: &str = "neoethos.canonical-trendbar-checkpoint-receipt.v1";
const MATRIX_RECEIPT_SCHEMA_V1: &str = "neoethos.canonical-trendbar-matrix-receipt.v1";
const CONTRACT_VERSION_V1: u16 = 1;
const PLAN_HASH_DOMAIN_V1: &[u8] = b"neoethos.canonical-trendbar-plan.v1\0";
const CHECKPOINT_HASH_DOMAIN_V1: &[u8] = b"neoethos.canonical-trendbar-checkpoint.v1\0";
const MATRIX_HASH_DOMAIN_V1: &[u8] = b"neoethos.canonical-trendbar-matrix.v1\0";
const PLAN_FILE_PREFIX_V1: &str = "ctp1-";
const CHECKPOINT_FILE_PREFIX_V1: &str = "ctc1-";
const MATRIX_FILE_PREFIX_V1: &str = "ctm1-";
const MAX_AUTHORITY_BYTES_V1: u64 = 32 * 1024 * 1024;
const MAX_SYMBOLS_V1: usize = 512;
static STAGING_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalTrendbarSymbolV1 {
    symbol_id: i64,
    symbol_name: String,
}

impl CanonicalTrendbarSymbolV1 {
    pub fn new(symbol_id: i64, symbol_name: impl Into<String>) -> Result<Self> {
        let symbol = Self {
            symbol_id,
            symbol_name: symbol_name.into(),
        };
        symbol.validate()?;
        Ok(symbol)
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    fn validate(&self) -> Result<()> {
        if self.symbol_id <= 0 {
            bail!("canonical trendbar symbol id must be positive");
        }
        if self.symbol_name.trim().is_empty()
            || self.symbol_name.len() > 64
            || self.symbol_name.chars().any(char::is_control)
        {
            bail!(
                "canonical trendbar symbol name {:?} is not one bounded broker symbol",
                self.symbol_name
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalTrendbarAcquisitionPlanV1 {
    environment: CTraderEnvironment,
    server: String,
    account_id: i64,
    from_ms: i64,
    to_ms_exclusive: i64,
    symbols: Vec<CanonicalTrendbarSymbolV1>,
    timeframes: Vec<CanonicalTimeframe>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalTrendbarAcquisitionPlanWireV1 {
    schema: String,
    version: u16,
    environment: String,
    server: String,
    account_id: i64,
    from_ms: i64,
    to_ms_exclusive: i64,
    paging_policy_id: String,
    symbols: Vec<CanonicalTrendbarSymbolV1>,
    timeframes: Vec<String>,
}

impl CanonicalTrendbarAcquisitionPlanV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        environment: CTraderEnvironment,
        server: impl Into<String>,
        account_id: i64,
        from_ms: i64,
        to_ms_exclusive: i64,
        mut symbols: Vec<CanonicalTrendbarSymbolV1>,
        mut timeframes: Vec<CanonicalTimeframe>,
    ) -> Result<Self> {
        symbols.sort_by(|left, right| {
            left.symbol_name
                .cmp(&right.symbol_name)
                .then(left.symbol_id.cmp(&right.symbol_id))
        });
        timeframes.sort_unstable();
        let plan = Self {
            environment,
            server: server.into(),
            account_id,
            from_ms,
            to_ms_exclusive,
            symbols,
            timeframes,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub const fn environment(&self) -> CTraderEnvironment {
        self.environment
    }

    pub fn server(&self) -> &str {
        &self.server
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn from_ms(&self) -> i64 {
        self.from_ms
    }

    pub const fn to_ms_exclusive(&self) -> i64 {
        self.to_ms_exclusive
    }

    pub const fn paging_policy_id(&self) -> &'static str {
        CANONICAL_TRENDBAR_PAGING_POLICY_V1
    }

    pub fn symbols(&self) -> &[CanonicalTrendbarSymbolV1] {
        &self.symbols
    }

    pub fn timeframes(&self) -> &[CanonicalTimeframe] {
        &self.timeframes
    }

    pub fn cell_count(&self) -> usize {
        self.symbols.len() * self.timeframes.len()
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(&self.to_wire()).context("encode canonical trendbar acquisition plan")
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        ensure_bounded(bytes, "canonical trendbar acquisition plan")?;
        let wire: CanonicalTrendbarAcquisitionPlanWireV1 =
            serde_json::from_slice(bytes).context("decode canonical trendbar acquisition plan")?;
        if wire.schema != PLAN_SCHEMA_V1 || wire.version != CONTRACT_VERSION_V1 {
            bail!(
                "unsupported canonical trendbar plan schema/version {:?}/{}",
                wire.schema,
                wire.version
            );
        }
        if wire.paging_policy_id != CANONICAL_TRENDBAR_PAGING_POLICY_V1 {
            bail!("canonical trendbar plan paging policy drifted");
        }
        let environment = parse_environment(&wire.environment)?;
        let original_symbols = wire.symbols.clone();
        let original_timeframes = wire
            .timeframes
            .iter()
            .map(|value| {
                value
                    .parse::<CanonicalTimeframe>()
                    .map_err(anyhow::Error::new)
            })
            .collect::<Result<Vec<_>>>()?;
        let plan = Self::new(
            environment,
            wire.server,
            wire.account_id,
            wire.from_ms,
            wire.to_ms_exclusive,
            wire.symbols,
            original_timeframes.clone(),
        )?;
        if plan.symbols != original_symbols || plan.timeframes != original_timeframes {
            bail!("canonical trendbar plan symbols/timeframes are not in canonical order");
        }
        if plan.to_json_bytes()? != bytes {
            bail!("canonical trendbar plan JSON is not canonically encoded");
        }
        Ok(plan)
    }

    fn identity_sha256(&self) -> Result<String> {
        Ok(domain_sha256(PLAN_HASH_DOMAIN_V1, &self.to_json_bytes()?))
    }

    fn expected_cell(
        &self,
        index: usize,
    ) -> Option<(CanonicalTrendbarSymbolV1, CanonicalTimeframe)> {
        if self.timeframes.is_empty() {
            return None;
        }
        let symbol = self.symbols.get(index / self.timeframes.len())?.clone();
        let timeframe = *self.timeframes.get(index % self.timeframes.len())?;
        Some((symbol, timeframe))
    }

    fn expected_identity(
        &self,
        symbol: &CanonicalTrendbarSymbolV1,
        timeframe: CanonicalTimeframe,
    ) -> Result<CanonicalDatasetIdentity> {
        CanonicalDatasetIdentity::ctrader(
            self.environment,
            &self.server,
            self.account_id,
            symbol.symbol_id,
            &symbol.symbol_name,
            timeframe,
            BarTimestampConvention::BarOpen,
        )
        .map_err(anyhow::Error::new)
    }

    fn validate(&self) -> Result<()> {
        if self.server.is_empty()
            || self.server.len() > 255
            || !self.server.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-')
            })
        {
            bail!("canonical trendbar server is not one bounded lowercase endpoint");
        }
        if self.account_id <= 0 {
            bail!("canonical trendbar account id must be positive");
        }
        if self.from_ms != CANONICAL_TRENDBAR_SERIES_FROM_MS_V1 {
            bail!("canonical trendbar v1 lower bound must be exactly 2016-01-01T00:00:00Z");
        }
        if self.to_ms_exclusive <= self.from_ms {
            bail!("canonical trendbar exclusive upper bound is missing or descending");
        }
        if self.symbols.is_empty() || self.symbols.len() > MAX_SYMBOLS_V1 {
            bail!("canonical trendbar plan symbol count is outside 1..={MAX_SYMBOLS_V1}");
        }
        if self.timeframes.is_empty() || self.timeframes.len() > CanonicalTimeframe::ALL.len() {
            bail!("canonical trendbar plan has no bounded canonical timeframe set");
        }
        let mut symbol_ids = HashSet::with_capacity(self.symbols.len());
        let mut symbol_names = HashSet::with_capacity(self.symbols.len());
        let mut previous_symbol: Option<&CanonicalTrendbarSymbolV1> = None;
        for symbol in &self.symbols {
            symbol.validate()?;
            if !symbol_ids.insert(symbol.symbol_id)
                || !symbol_names.insert(symbol.symbol_name.as_str())
            {
                bail!("canonical trendbar plan repeats a symbol id or name");
            }
            if previous_symbol.is_some_and(|previous| {
                (previous.symbol_name.as_str(), previous.symbol_id)
                    >= (symbol.symbol_name.as_str(), symbol.symbol_id)
            }) {
                bail!("canonical trendbar plan symbols are not canonically ordered");
            }
            previous_symbol = Some(symbol);
        }
        let mut previous_timeframe = None;
        for timeframe in &self.timeframes {
            if previous_timeframe.is_some_and(|previous| previous >= *timeframe) {
                bail!("canonical trendbar plan timeframes are duplicate or out of order");
            }
            previous_timeframe = Some(*timeframe);
        }
        Ok(())
    }

    fn to_wire(&self) -> CanonicalTrendbarAcquisitionPlanWireV1 {
        CanonicalTrendbarAcquisitionPlanWireV1 {
            schema: PLAN_SCHEMA_V1.to_owned(),
            version: CONTRACT_VERSION_V1,
            environment: self.environment.as_str().to_owned(),
            server: self.server.clone(),
            account_id: self.account_id,
            from_ms: self.from_ms,
            to_ms_exclusive: self.to_ms_exclusive,
            paging_policy_id: CANONICAL_TRENDBAR_PAGING_POLICY_V1.to_owned(),
            symbols: self.symbols.clone(),
            timeframes: self
                .timeframes
                .iter()
                .map(|timeframe| timeframe.as_str().to_owned())
                .collect(),
        }
    }
}

macro_rules! content_receipt {
    ($name:ident, $schema:expr) => {
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name {
            schema: String,
            version: u16,
            sha256: String,
        }

        impl $name {
            fn new(sha256: String) -> Result<Self> {
                let receipt = Self {
                    schema: $schema.to_owned(),
                    version: CONTRACT_VERSION_V1,
                    sha256,
                };
                receipt.validate()?;
                Ok(receipt)
            }

            pub fn from_sha256(sha256: String) -> Result<Self> {
                Self::new(sha256)
            }

            pub fn sha256(&self) -> &str {
                &self.sha256
            }

            pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
                self.validate()?;
                serde_json::to_vec(self).context("encode canonical trendbar content receipt")
            }

            pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
                ensure_bounded(bytes, "canonical trendbar content receipt")?;
                let receipt: Self = serde_json::from_slice(bytes)
                    .context("decode canonical trendbar content receipt")?;
                receipt.validate()?;
                if receipt.to_json_bytes()? != bytes {
                    bail!("canonical trendbar content receipt JSON is not canonical");
                }
                Ok(receipt)
            }

            fn validate(&self) -> Result<()> {
                if self.schema != $schema || self.version != CONTRACT_VERSION_V1 {
                    bail!("unsupported canonical trendbar content receipt schema/version");
                }
                validate_sha256("canonical trendbar content receipt", &self.sha256)
            }
        }
    };
}

content_receipt!(CanonicalTrendbarPlanReceiptV1, PLAN_RECEIPT_SCHEMA_V1);
content_receipt!(
    CanonicalTrendbarCheckpointReceiptV1,
    CHECKPOINT_RECEIPT_SCHEMA_V1
);
content_receipt!(CanonicalTrendbarMatrixReceiptV1, MATRIX_RECEIPT_SCHEMA_V1);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalTrendbarAcquisitionCellV1 {
    selected_generation: SelectedDatasetGenerationV1,
}

impl CanonicalTrendbarAcquisitionCellV1 {
    pub fn new(selected_generation: SelectedDatasetGenerationV1) -> Result<Self> {
        selected_generation.validate()?;
        Ok(Self {
            selected_generation,
        })
    }

    pub const fn selected_generation(&self) -> &SelectedDatasetGenerationV1 {
        &self.selected_generation
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalTrendbarAcquisitionCheckpointV1 {
    plan_sha256: String,
    previous_checkpoint_sha256: Option<String>,
    completed_cells: Vec<CanonicalTrendbarAcquisitionCellV1>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalTrendbarAcquisitionCheckpointWireV1 {
    schema: String,
    version: u16,
    plan_sha256: String,
    previous_checkpoint_sha256: Option<String>,
    completed_cells: Vec<CanonicalTrendbarAcquisitionCellV1>,
}

impl CanonicalTrendbarAcquisitionCheckpointV1 {
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }

    pub fn previous_checkpoint_sha256(&self) -> Option<&str> {
        self.previous_checkpoint_sha256.as_deref()
    }

    pub fn completed_cells(&self) -> &[CanonicalTrendbarAcquisitionCellV1] {
        &self.completed_cells
    }

    pub fn next_cell(
        &self,
        plan: &CanonicalTrendbarAcquisitionPlanV1,
    ) -> Result<Option<(CanonicalTrendbarSymbolV1, CanonicalTimeframe)>> {
        if plan.identity_sha256()? != self.plan_sha256 {
            bail!("checkpoint belongs to a different canonical trendbar plan");
        }
        Ok(plan.expected_cell(self.completed_cells.len()))
    }

    fn new(
        plan_sha256: String,
        previous_checkpoint_sha256: Option<String>,
        completed_cells: Vec<CanonicalTrendbarAcquisitionCellV1>,
    ) -> Result<Self> {
        let checkpoint = Self {
            plan_sha256,
            previous_checkpoint_sha256,
            completed_cells,
        };
        checkpoint.validate_basic()?;
        Ok(checkpoint)
    }

    fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate_basic()?;
        serde_json::to_vec(&CanonicalTrendbarAcquisitionCheckpointWireV1 {
            schema: CHECKPOINT_SCHEMA_V1.to_owned(),
            version: CONTRACT_VERSION_V1,
            plan_sha256: self.plan_sha256.clone(),
            previous_checkpoint_sha256: self.previous_checkpoint_sha256.clone(),
            completed_cells: self.completed_cells.clone(),
        })
        .context("encode canonical trendbar checkpoint")
    }

    fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        ensure_bounded(bytes, "canonical trendbar checkpoint")?;
        let wire: CanonicalTrendbarAcquisitionCheckpointWireV1 =
            serde_json::from_slice(bytes).context("decode canonical trendbar checkpoint")?;
        if wire.schema != CHECKPOINT_SCHEMA_V1 || wire.version != CONTRACT_VERSION_V1 {
            bail!("unsupported canonical trendbar checkpoint schema/version");
        }
        let checkpoint = Self::new(
            wire.plan_sha256,
            wire.previous_checkpoint_sha256,
            wire.completed_cells,
        )?;
        if checkpoint.to_json_bytes()? != bytes {
            bail!("canonical trendbar checkpoint JSON is not canonical");
        }
        Ok(checkpoint)
    }

    fn validate_basic(&self) -> Result<()> {
        validate_sha256("canonical trendbar checkpoint plan", &self.plan_sha256)?;
        if let Some(previous) = &self.previous_checkpoint_sha256 {
            validate_sha256("canonical trendbar previous checkpoint", previous)?;
        }
        for cell in &self.completed_cells {
            cell.selected_generation.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalTrendbarMatrixV1 {
    plan_sha256: String,
    checkpoint_sha256: String,
    series: Vec<CanonicalDatasetSeriesReceiptV1>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalTrendbarMatrixWireV1 {
    schema: String,
    version: u16,
    plan_sha256: String,
    checkpoint_sha256: String,
    series: Vec<CanonicalDatasetSeriesReceiptV1>,
}

impl CanonicalTrendbarMatrixV1 {
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }

    pub fn checkpoint_sha256(&self) -> &str {
        &self.checkpoint_sha256
    }

    pub fn series(&self) -> &[CanonicalDatasetSeriesReceiptV1] {
        &self.series
    }

    fn new(
        plan_sha256: String,
        checkpoint_sha256: String,
        series: Vec<CanonicalDatasetSeriesReceiptV1>,
    ) -> Result<Self> {
        let matrix = Self {
            plan_sha256,
            checkpoint_sha256,
            series,
        };
        matrix.validate_basic()?;
        Ok(matrix)
    }

    fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate_basic()?;
        serde_json::to_vec(&CanonicalTrendbarMatrixWireV1 {
            schema: MATRIX_SCHEMA_V1.to_owned(),
            version: CONTRACT_VERSION_V1,
            plan_sha256: self.plan_sha256.clone(),
            checkpoint_sha256: self.checkpoint_sha256.clone(),
            series: self.series.clone(),
        })
        .context("encode canonical trendbar matrix")
    }

    fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        ensure_bounded(bytes, "canonical trendbar matrix")?;
        let wire: CanonicalTrendbarMatrixWireV1 =
            serde_json::from_slice(bytes).context("decode canonical trendbar matrix")?;
        if wire.schema != MATRIX_SCHEMA_V1 || wire.version != CONTRACT_VERSION_V1 {
            bail!("unsupported canonical trendbar matrix schema/version");
        }
        let matrix = Self::new(wire.plan_sha256, wire.checkpoint_sha256, wire.series)?;
        if matrix.to_json_bytes()? != bytes {
            bail!("canonical trendbar matrix JSON is not canonical");
        }
        Ok(matrix)
    }

    fn validate_basic(&self) -> Result<()> {
        validate_sha256("canonical trendbar matrix plan", &self.plan_sha256)?;
        validate_sha256(
            "canonical trendbar matrix checkpoint",
            &self.checkpoint_sha256,
        )?;
        if self.series.is_empty() {
            bail!("canonical trendbar matrix has no symbol series");
        }
        for series in &self.series {
            series.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CanonicalTrendbarAcquisitionStoreV1 {
    root: PathBuf,
}

impl CanonicalTrendbarAcquisitionStoreV1 {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn plan_path(&self, receipt: &CanonicalTrendbarPlanReceiptV1) -> PathBuf {
        self.content_path(PLAN_FILE_PREFIX_V1, receipt.sha256())
    }

    pub fn checkpoint_path(&self, receipt: &CanonicalTrendbarCheckpointReceiptV1) -> PathBuf {
        self.content_path(CHECKPOINT_FILE_PREFIX_V1, receipt.sha256())
    }

    pub fn matrix_path(&self, receipt: &CanonicalTrendbarMatrixReceiptV1) -> PathBuf {
        self.content_path(MATRIX_FILE_PREFIX_V1, receipt.sha256())
    }

    pub fn publish_plan(
        &self,
        plan: &CanonicalTrendbarAcquisitionPlanV1,
    ) -> Result<CanonicalTrendbarPlanReceiptV1> {
        let bytes = plan.to_json_bytes()?;
        let receipt =
            CanonicalTrendbarPlanReceiptV1::new(domain_sha256(PLAN_HASH_DOMAIN_V1, &bytes))?;
        self.publish_content(&self.plan_path(&receipt), PLAN_FILE_PREFIX_V1, &bytes)?;
        if self.open_plan(&receipt)? != *plan {
            bail!("published canonical trendbar plan did not reopen exactly");
        }
        Ok(receipt)
    }

    pub fn open_plan(
        &self,
        receipt: &CanonicalTrendbarPlanReceiptV1,
    ) -> Result<CanonicalTrendbarAcquisitionPlanV1> {
        receipt.validate()?;
        let bytes = self.read_content(&self.plan_path(receipt))?;
        let actual = domain_sha256(PLAN_HASH_DOMAIN_V1, &bytes);
        if actual != receipt.sha256 {
            bail!("canonical trendbar plan content digest differs from its exact receipt");
        }
        let plan = CanonicalTrendbarAcquisitionPlanV1::from_json_bytes(&bytes)?;
        if plan.identity_sha256()? != receipt.sha256 {
            bail!("canonical trendbar plan identity differs from its exact receipt");
        }
        Ok(plan)
    }

    pub fn publish_checkpoint(
        &self,
        data_root: &Path,
        plan_receipt: &CanonicalTrendbarPlanReceiptV1,
        previous: Option<&CanonicalTrendbarCheckpointReceiptV1>,
        completed_cells: Vec<CanonicalTrendbarAcquisitionCellV1>,
    ) -> Result<CanonicalTrendbarCheckpointReceiptV1> {
        let plan = self.open_plan(plan_receipt)?;
        let checkpoint = CanonicalTrendbarAcquisitionCheckpointV1::new(
            plan_receipt.sha256.clone(),
            previous.map(|receipt| receipt.sha256.clone()),
            completed_cells,
        )?;
        self.validate_checkpoint(data_root, plan_receipt, &plan, &checkpoint)?;
        let bytes = checkpoint.to_json_bytes()?;
        let receipt = CanonicalTrendbarCheckpointReceiptV1::new(domain_sha256(
            CHECKPOINT_HASH_DOMAIN_V1,
            &bytes,
        ))?;
        self.publish_content(
            &self.checkpoint_path(&receipt),
            CHECKPOINT_FILE_PREFIX_V1,
            &bytes,
        )?;
        self.open_checkpoint(data_root, plan_receipt, &receipt)?;
        Ok(receipt)
    }

    pub fn open_checkpoint(
        &self,
        data_root: &Path,
        plan_receipt: &CanonicalTrendbarPlanReceiptV1,
        receipt: &CanonicalTrendbarCheckpointReceiptV1,
    ) -> Result<CanonicalTrendbarAcquisitionCheckpointV1> {
        receipt.validate()?;
        let plan = self.open_plan(plan_receipt)?;
        let checkpoint = self.read_checkpoint(receipt)?;
        self.validate_checkpoint(data_root, plan_receipt, &plan, &checkpoint)?;
        Ok(checkpoint)
    }

    pub fn publish_matrix(
        &self,
        data_root: &Path,
        plan_receipt: &CanonicalTrendbarPlanReceiptV1,
        checkpoint_receipt: &CanonicalTrendbarCheckpointReceiptV1,
    ) -> Result<CanonicalTrendbarMatrixReceiptV1> {
        let plan = self.open_plan(plan_receipt)?;
        let checkpoint = self.open_checkpoint(data_root, plan_receipt, checkpoint_receipt)?;
        let matrix = build_matrix(plan_receipt, checkpoint_receipt, &plan, &checkpoint)?;
        let bytes = matrix.to_json_bytes()?;
        let receipt =
            CanonicalTrendbarMatrixReceiptV1::new(domain_sha256(MATRIX_HASH_DOMAIN_V1, &bytes))?;
        self.publish_content(&self.matrix_path(&receipt), MATRIX_FILE_PREFIX_V1, &bytes)?;
        self.open_matrix(data_root, plan_receipt, &receipt)?;
        Ok(receipt)
    }

    pub fn open_matrix(
        &self,
        data_root: &Path,
        plan_receipt: &CanonicalTrendbarPlanReceiptV1,
        receipt: &CanonicalTrendbarMatrixReceiptV1,
    ) -> Result<CanonicalTrendbarMatrixV1> {
        receipt.validate()?;
        let bytes = self.read_content(&self.matrix_path(receipt))?;
        if domain_sha256(MATRIX_HASH_DOMAIN_V1, &bytes) != receipt.sha256 {
            bail!("canonical trendbar matrix content digest differs from its exact receipt");
        }
        let matrix = CanonicalTrendbarMatrixV1::from_json_bytes(&bytes)?;
        if matrix.plan_sha256 != plan_receipt.sha256 {
            bail!("canonical trendbar matrix belongs to a different plan");
        }
        let checkpoint_receipt =
            CanonicalTrendbarCheckpointReceiptV1::new(matrix.checkpoint_sha256.clone())?;
        let plan = self.open_plan(plan_receipt)?;
        let checkpoint = self.open_checkpoint(data_root, plan_receipt, &checkpoint_receipt)?;
        let expected = build_matrix(plan_receipt, &checkpoint_receipt, &plan, &checkpoint)?;
        if matrix != expected {
            bail!("canonical trendbar matrix differs from its exact complete checkpoint");
        }
        Ok(matrix)
    }

    fn validate_checkpoint(
        &self,
        data_root: &Path,
        plan_receipt: &CanonicalTrendbarPlanReceiptV1,
        plan: &CanonicalTrendbarAcquisitionPlanV1,
        checkpoint: &CanonicalTrendbarAcquisitionCheckpointV1,
    ) -> Result<()> {
        if checkpoint.plan_sha256 != plan_receipt.sha256 {
            bail!("canonical trendbar checkpoint belongs to a different plan");
        }
        if checkpoint.completed_cells.len() > plan.cell_count() {
            bail!("canonical trendbar checkpoint contains extra cells");
        }
        for (index, cell) in checkpoint.completed_cells.iter().enumerate() {
            let (symbol, timeframe) = plan
                .expected_cell(index)
                .context("checkpoint cell is outside the exact plan matrix")?;
            validate_completed_cell(data_root, plan, &symbol, timeframe, cell)?;
        }
        if let Some(previous_sha256) = &checkpoint.previous_checkpoint_sha256 {
            let previous_receipt =
                CanonicalTrendbarCheckpointReceiptV1::new(previous_sha256.clone())?;
            let previous = self.read_checkpoint(&previous_receipt)?;
            if previous.plan_sha256 != plan_receipt.sha256 {
                bail!("previous canonical trendbar checkpoint belongs to a different plan");
            }
            if checkpoint.completed_cells.len() <= previous.completed_cells.len()
                || !checkpoint
                    .completed_cells
                    .starts_with(&previous.completed_cells)
            {
                bail!(
                    "canonical trendbar checkpoint does not strictly extend its exact predecessor"
                );
            }
        }
        Ok(())
    }

    fn read_checkpoint(
        &self,
        receipt: &CanonicalTrendbarCheckpointReceiptV1,
    ) -> Result<CanonicalTrendbarAcquisitionCheckpointV1> {
        let bytes = self.read_content(&self.checkpoint_path(receipt))?;
        if domain_sha256(CHECKPOINT_HASH_DOMAIN_V1, &bytes) != receipt.sha256 {
            bail!("canonical trendbar checkpoint content digest differs from its exact receipt");
        }
        CanonicalTrendbarAcquisitionCheckpointV1::from_json_bytes(&bytes)
    }

    fn content_path(&self, prefix: &str, sha256: &str) -> PathBuf {
        self.root.join(format!("{prefix}{sha256}.json"))
    }

    fn publish_content(&self, final_path: &Path, prefix: &str, bytes: &[u8]) -> Result<()> {
        self.ensure_safe_root()?;
        match fs::symlink_metadata(final_path) {
            Ok(_) => {
                if self.read_content(final_path)? != bytes {
                    bail!(
                        "content-addressed canonical trendbar collision at {}",
                        final_path.display()
                    );
                }
                return Ok(());
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("inspect canonical trendbar content path"),
        }

        let staging = self.create_staging_path(prefix)?;
        let publish = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&staging)
                .with_context(|| format!("create staging file {}", staging.display()))?;
            file.write_all(bytes)
                .with_context(|| format!("write staging file {}", staging.display()))?;
            file.sync_all()
                .with_context(|| format!("fsync staging file {}", staging.display()))?;
            drop(file);
            match fs::rename(&staging, final_path) {
                Ok(()) => Ok(()),
                Err(error) if final_path.exists() => {
                    let _ = fs::remove_file(&staging);
                    if self.read_content(final_path)? == bytes {
                        Ok(())
                    } else {
                        Err(error).context("content-addressed publication race collided")
                    }
                }
                Err(error) => Err(error).with_context(|| {
                    format!(
                        "publish immutable canonical trendbar content {}",
                        final_path.display()
                    )
                }),
            }
        })();
        if publish.is_err() {
            cleanup_staging(&self.root, &staging, prefix);
        }
        publish?;
        if self.read_content(final_path)? != bytes {
            bail!("published canonical trendbar content did not reopen byte-exact");
        }
        Ok(())
    }

    fn read_content(&self, path: &Path) -> Result<Vec<u8>> {
        self.require_existing_safe_root()?;
        let metadata = fs::symlink_metadata(path)
            .with_context(|| format!("inspect immutable content {}", path.display()))?;
        reject_link_or_reparse(path, &metadata)?;
        if !metadata.is_file() || metadata.len() > MAX_AUTHORITY_BYTES_V1 {
            bail!("canonical trendbar content is not one bounded regular file");
        }
        let file = File::open(path)
            .with_context(|| format!("open immutable content {}", path.display()))?;
        let mut reader = file.take(MAX_AUTHORITY_BYTES_V1 + 1);
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        reader
            .read_to_end(&mut bytes)
            .with_context(|| format!("read immutable content {}", path.display()))?;
        ensure_bounded(&bytes, "canonical trendbar immutable content")?;
        Ok(bytes)
    }

    fn ensure_safe_root(&self) -> Result<()> {
        match fs::symlink_metadata(&self.root) {
            Ok(metadata) => {
                reject_link_or_reparse(&self.root, &metadata)?;
                if !metadata.is_dir() {
                    bail!("canonical trendbar authority root is not a directory");
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&self.root).with_context(|| {
                    format!(
                        "create canonical trendbar authority root {}",
                        self.root.display()
                    )
                })?;
                let metadata = fs::symlink_metadata(&self.root)?;
                reject_link_or_reparse(&self.root, &metadata)?;
                if !metadata.is_dir() {
                    bail!("created canonical trendbar authority root is not a directory");
                }
            }
            Err(error) => return Err(error).context("inspect canonical trendbar authority root"),
        }
        Ok(())
    }

    fn require_existing_safe_root(&self) -> Result<()> {
        let metadata = fs::symlink_metadata(&self.root).with_context(|| {
            format!(
                "inspect existing canonical trendbar authority root {}",
                self.root.display()
            )
        })?;
        reject_link_or_reparse(&self.root, &metadata)?;
        if !metadata.is_dir() {
            bail!("canonical trendbar authority root is not a directory");
        }
        Ok(())
    }

    fn create_staging_path(&self, prefix: &str) -> Result<PathBuf> {
        for _ in 0..32 {
            let clock = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("system clock is before Unix epoch")?
                .as_nanos();
            let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = self.root.join(format!(
                ".{prefix}staging-{}-{clock}-{nonce}.tmp",
                std::process::id()
            ));
            if !path.exists() {
                return Ok(path);
            }
        }
        bail!("cannot allocate canonical trendbar staging path")
    }
}

fn validate_completed_cell(
    data_root: &Path,
    plan: &CanonicalTrendbarAcquisitionPlanV1,
    symbol: &CanonicalTrendbarSymbolV1,
    timeframe: CanonicalTimeframe,
    cell: &CanonicalTrendbarAcquisitionCellV1,
) -> Result<()> {
    let expected_identity = plan.expected_identity(symbol, timeframe)?;
    if cell.selected_generation.identity() != &expected_identity {
        bail!("completed canonical trendbar cell differs from the exact plan identity/order");
    }
    let (manifest, lease) = open_exact_dataset_generation(data_root, &cell.selected_generation)
        .context("reopen exact completed canonical trendbar generation")?;
    let provenance = CTraderTrendbarProvenanceV1::from_envelope(manifest.provenance())?;
    if provenance.dataset_identity() != &expected_identity
        || provenance.requested_range_ms() != (plan.from_ms, plan.to_ms_exclusive)
        || provenance.row_count() != manifest.row_count()
    {
        bail!("completed canonical trendbar provenance differs from the exact plan/window");
    }
    drop(lease);
    Ok(())
}

fn build_matrix(
    plan_receipt: &CanonicalTrendbarPlanReceiptV1,
    checkpoint_receipt: &CanonicalTrendbarCheckpointReceiptV1,
    plan: &CanonicalTrendbarAcquisitionPlanV1,
    checkpoint: &CanonicalTrendbarAcquisitionCheckpointV1,
) -> Result<CanonicalTrendbarMatrixV1> {
    if checkpoint.completed_cells.len() != plan.cell_count() {
        bail!("canonical trendbar matrix requires every exact plan cell");
    }
    let mut series = Vec::with_capacity(plan.symbols.len());
    for symbol_index in 0..plan.symbols.len() {
        let start = symbol_index * plan.timeframes.len();
        let end = start + plan.timeframes.len();
        let direct = checkpoint.completed_cells[start..end]
            .iter()
            .map(|cell| cell.selected_generation.clone())
            .collect::<Vec<_>>();
        let anchor = direct
            .first()
            .cloned()
            .context("canonical trendbar symbol has no anchor timeframe")?;
        series.push(CanonicalDatasetSeriesReceiptV1::new(anchor, direct)?);
    }
    CanonicalTrendbarMatrixV1::new(
        plan_receipt.sha256.clone(),
        checkpoint_receipt.sha256.clone(),
        series,
    )
}

fn parse_environment(value: &str) -> Result<CTraderEnvironment> {
    match value {
        "demo" => Ok(CTraderEnvironment::Demo),
        "live" => Ok(CTraderEnvironment::Live),
        _ => bail!("unsupported canonical trendbar cTrader environment {value:?}"),
    }
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} is not canonical lowercase SHA-256 hex");
    }
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn ensure_bounded(bytes: &[u8], label: &str) -> Result<()> {
    if bytes.len() as u64 > MAX_AUTHORITY_BYTES_V1 {
        bail!("{label} exceeds the {MAX_AUTHORITY_BYTES_V1}-byte limit");
    }
    Ok(())
}

fn cleanup_staging(root: &Path, staging: &Path, prefix: &str) {
    let expected_prefix = format!(".{prefix}staging-");
    if staging.parent() == Some(root)
        && staging
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(&expected_prefix))
    {
        let _ = fs::remove_file(staging);
    }
}

fn reject_link_or_reparse(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    if metadata.file_type().is_symlink() {
        bail!(
            "refusing symlink at canonical trendbar authority path {}",
            path.display()
        );
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            bail!(
                "refusing reparse point at canonical trendbar authority path {}",
                path.display()
            );
        }
    }
    Ok(())
}
