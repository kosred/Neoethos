//! Persisted pre-touch quote-coverage handshake for the bounded V1 lane.
//!
//! This module deliberately stops at `QuoteCoverageReady`: it never spends the
//! single OOS touch and never evaluates a signal. A later, separately reviewed
//! continuation may consume the exact ready evidence once.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::{self, Write};

use crate::journal::OosWindow;
use crate::session::{DatasetReceiptV1, SessionId, SweepId};

pub const QUOTE_COVERAGE_REQUEST_SCHEMA_V1: &str =
    "neoethos.autoresearch.quote-coverage-request.v1";
pub const QUOTE_COVERAGE_READY_SCHEMA_V1: &str = "neoethos.autoresearch.quote-coverage-ready.v1";
pub const MAX_QUOTE_COVERAGE_REQUEST_BYTES_V1: usize = 1024 * 1024;

const REQUEST_ID_DOMAIN_V1: &[u8] = b"neoethos.autoresearch.quote-coverage-request.identity.v1\0";
const READY_ID_DOMAIN_V1: &[u8] = b"neoethos.autoresearch.quote-coverage-ready.identity.v1\0";
const MAX_IDENTITY_TEXT_BYTES_V1: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteCoverageErrorCodeV1 {
    InvalidIdentity,
    UnsupportedV1Shape,
    RequestTooLarge,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuoteCoverageErrorV1 {
    code: QuoteCoverageErrorCodeV1,
    detail: String,
}

impl QuoteCoverageErrorV1 {
    fn new(code: QuoteCoverageErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> QuoteCoverageErrorCodeV1 {
        self.code
    }
}

impl fmt::Display for QuoteCoverageErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for QuoteCoverageErrorV1 {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuoteCoverageRequestV1 {
    schema: String,
    session_id: SessionId,
    window: OosWindow,
    sweep: SweepId,
    slot: usize,
    candidate_config_hash: String,
    dataset_receipt: DatasetReceiptV1,
    portfolio_identity_sha256: String,
    effective_search_config_hash: String,
    batch_count: usize,
    gene_count: usize,
    fixed_stop_only: bool,
    request_identity_sha256: String,
}

#[derive(Serialize)]
struct QuoteCoverageRequestIdentityMaterialV1<'a> {
    schema: &'a str,
    session_id: &'a SessionId,
    window: OosWindow,
    sweep: SweepId,
    slot: usize,
    candidate_config_hash: &'a str,
    dataset_receipt: &'a DatasetReceiptV1,
    portfolio_identity_sha256: &'a str,
    effective_search_config_hash: &'a str,
    batch_count: usize,
    gene_count: usize,
    fixed_stop_only: bool,
}

impl QuoteCoverageRequestV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_id: SessionId,
        window: OosWindow,
        sweep: SweepId,
        slot: usize,
        candidate_config_hash: String,
        dataset_receipt: DatasetReceiptV1,
        portfolio_identity_sha256: String,
        effective_search_config_hash: String,
        batch_count: usize,
        gene_count: usize,
        fixed_stop_only: bool,
    ) -> Result<Self, QuoteCoverageErrorV1> {
        let mut request = Self {
            schema: QUOTE_COVERAGE_REQUEST_SCHEMA_V1.to_owned(),
            session_id,
            window,
            sweep,
            slot,
            candidate_config_hash,
            dataset_receipt,
            portfolio_identity_sha256,
            effective_search_config_hash,
            batch_count,
            gene_count,
            fixed_stop_only,
            request_identity_sha256: String::new(),
        };
        request.validate_shape_without_identity()?;
        request.request_identity_sha256 = request.compute_identity_sha256()?;
        request.validate()?;
        Ok(request)
    }

    pub(crate) fn from_candidate(
        session_id: &SessionId,
        window: OosWindow,
        portfolio: &crate::runner::PromotionPortfolio,
    ) -> Result<Self, QuoteCoverageErrorV1> {
        let portfolio_identity_sha256 =
            neoethos_search::canonical_locked_portfolio_identity_sha256_v1(portfolio).map_err(
                |error| {
                    QuoteCoverageErrorV1::new(
                        QuoteCoverageErrorCodeV1::InvalidIdentity,
                        format!("computing exact finalist portfolio identity: {error}"),
                    )
                },
            )?;
        let effective_search_config_hash = portfolio
            .batch_bindings
            .first()
            .map(|binding| binding.search_config_hash.clone())
            .unwrap_or_default();
        let fixed_stop_only = portfolio.batch_bindings.iter().all(|binding| {
            binding.genes.iter().all(|tagged| {
                tagged.gene.stop_vol_mult.is_finite() && tagged.gene.stop_vol_mult == 0.0
            })
        });
        Self::new(
            session_id.clone(),
            window,
            portfolio.sweep,
            portfolio.slot,
            portfolio.config_hash.clone(),
            portfolio.dataset_receipt.clone(),
            portfolio_identity_sha256,
            effective_search_config_hash,
            portfolio.batch_count,
            portfolio.gene_count,
            fixed_stop_only,
        )
    }

    pub(crate) fn validate(&self) -> Result<(), QuoteCoverageErrorV1> {
        self.validate_shape_without_identity()?;
        let expected = self.compute_identity_sha256()?;
        if self.request_identity_sha256 != expected {
            return Err(QuoteCoverageErrorV1::new(
                QuoteCoverageErrorCodeV1::InvalidIdentity,
                format!(
                    "quote-coverage request identity {} recomputes to {expected}",
                    self.request_identity_sha256
                ),
            ));
        }
        Ok(())
    }

    fn validate_shape_without_identity(&self) -> Result<(), QuoteCoverageErrorV1> {
        if self.schema != QUOTE_COVERAGE_REQUEST_SCHEMA_V1 {
            return Err(QuoteCoverageErrorV1::new(
                QuoteCoverageErrorCodeV1::UnsupportedV1Shape,
                format!(
                    "quote-coverage request schema {:?} is not {QUOTE_COVERAGE_REQUEST_SCHEMA_V1}",
                    self.schema
                ),
            ));
        }
        if self.batch_count != 1 || self.gene_count == 0 || !self.fixed_stop_only {
            return Err(QuoteCoverageErrorV1::new(
                QuoteCoverageErrorCodeV1::UnsupportedV1Shape,
                "quote-coverage V1 requires exactly one nonempty batch with fixed stops",
            ));
        }
        if self.window.start_ms > self.window.end_ms
            || self.dataset_receipt.oos_window != self.window
        {
            return Err(QuoteCoverageErrorV1::new(
                QuoteCoverageErrorCodeV1::UnsupportedV1Shape,
                "quote-coverage window is inverted or detached from its dataset receipt",
            ));
        }
        self.dataset_receipt.validate().map_err(|error| {
            QuoteCoverageErrorV1::new(
                QuoteCoverageErrorCodeV1::UnsupportedV1Shape,
                format!("invalid quote-coverage dataset receipt: {error:#}"),
            )
        })?;
        validate_bounded_text("candidate_config_hash", &self.candidate_config_hash)?;
        validate_bounded_text(
            "effective_search_config_hash",
            &self.effective_search_config_hash,
        )?;
        validate_lowerhex_sha256("portfolio_identity_sha256", &self.portfolio_identity_sha256)?;
        let count = compact_json_count(self)?;
        if count > MAX_QUOTE_COVERAGE_REQUEST_BYTES_V1 {
            return Err(QuoteCoverageErrorV1::new(
                QuoteCoverageErrorCodeV1::RequestTooLarge,
                format!(
                    "quote-coverage request is {count} bytes, above the {}-byte V1 cap",
                    MAX_QUOTE_COVERAGE_REQUEST_BYTES_V1
                ),
            ));
        }
        Ok(())
    }

    fn compute_identity_sha256(&self) -> Result<String, QuoteCoverageErrorV1> {
        let material = QuoteCoverageRequestIdentityMaterialV1 {
            schema: &self.schema,
            session_id: &self.session_id,
            window: self.window,
            sweep: self.sweep,
            slot: self.slot,
            candidate_config_hash: &self.candidate_config_hash,
            dataset_receipt: &self.dataset_receipt,
            portfolio_identity_sha256: &self.portfolio_identity_sha256,
            effective_search_config_hash: &self.effective_search_config_hash,
            batch_count: self.batch_count,
            gene_count: self.gene_count,
            fixed_stop_only: self.fixed_stop_only,
        };
        hash_compact_json(REQUEST_ID_DOMAIN_V1, &material)
    }

    pub fn compact_json_byte_count(&self) -> usize {
        compact_json_count(self).unwrap_or(usize::MAX)
    }

    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub fn window(&self) -> OosWindow {
        self.window
    }

    pub fn sweep(&self) -> SweepId {
        self.sweep
    }

    pub fn slot(&self) -> usize {
        self.slot
    }

    pub fn candidate_config_hash(&self) -> &str {
        &self.candidate_config_hash
    }

    pub fn dataset_receipt(&self) -> &DatasetReceiptV1 {
        &self.dataset_receipt
    }

    pub fn portfolio_identity_sha256(&self) -> &str {
        &self.portfolio_identity_sha256
    }

    pub fn effective_search_config_hash(&self) -> &str {
        &self.effective_search_config_hash
    }

    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    pub fn gene_count(&self) -> usize {
        self.gene_count
    }

    pub fn fixed_stop_only(&self) -> bool {
        self.fixed_stop_only
    }

    pub fn request_identity_sha256(&self) -> &str {
        &self.request_identity_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuoteCoverageReadyV1 {
    schema: String,
    request_identity_sha256: String,
    session_id: SessionId,
    sweep: SweepId,
    slot: usize,
    candidate_config_hash: String,
    coverage_receipt_sha256: String,
    ready_identity_sha256: String,
}

#[derive(Serialize)]
struct QuoteCoverageReadyIdentityMaterialV1<'a> {
    schema: &'a str,
    request_identity_sha256: &'a str,
    session_id: &'a SessionId,
    sweep: SweepId,
    slot: usize,
    candidate_config_hash: &'a str,
    coverage_receipt_sha256: &'a str,
}

impl QuoteCoverageReadyV1 {
    pub fn new(
        request: &QuoteCoverageRequestV1,
        coverage_receipt_sha256: String,
    ) -> Result<Self, QuoteCoverageErrorV1> {
        request.validate()?;
        validate_lowerhex_sha256("coverage_receipt_sha256", &coverage_receipt_sha256)?;
        let mut ready = Self {
            schema: QUOTE_COVERAGE_READY_SCHEMA_V1.to_owned(),
            request_identity_sha256: request.request_identity_sha256.clone(),
            session_id: request.session_id.clone(),
            sweep: request.sweep,
            slot: request.slot,
            candidate_config_hash: request.candidate_config_hash.clone(),
            coverage_receipt_sha256,
            ready_identity_sha256: String::new(),
        };
        ready.ready_identity_sha256 = ready.compute_identity_sha256()?;
        ready.validate_against(request)?;
        Ok(ready)
    }

    pub(crate) fn validate_against(
        &self,
        request: &QuoteCoverageRequestV1,
    ) -> Result<(), QuoteCoverageErrorV1> {
        request.validate()?;
        if self.schema != QUOTE_COVERAGE_READY_SCHEMA_V1
            || self.request_identity_sha256 != request.request_identity_sha256
            || self.session_id != request.session_id
            || self.sweep != request.sweep
            || self.slot != request.slot
            || self.candidate_config_hash != request.candidate_config_hash
        {
            return Err(QuoteCoverageErrorV1::new(
                QuoteCoverageErrorCodeV1::InvalidIdentity,
                "Ready evidence belongs to a different quote-coverage request",
            ));
        }
        validate_lowerhex_sha256("coverage_receipt_sha256", &self.coverage_receipt_sha256)?;
        let expected = self.compute_identity_sha256()?;
        if self.ready_identity_sha256 != expected {
            return Err(QuoteCoverageErrorV1::new(
                QuoteCoverageErrorCodeV1::InvalidIdentity,
                format!(
                    "quote-coverage Ready identity {} recomputes to {expected}",
                    self.ready_identity_sha256
                ),
            ));
        }
        Ok(())
    }

    fn compute_identity_sha256(&self) -> Result<String, QuoteCoverageErrorV1> {
        let material = QuoteCoverageReadyIdentityMaterialV1 {
            schema: &self.schema,
            request_identity_sha256: &self.request_identity_sha256,
            session_id: &self.session_id,
            sweep: self.sweep,
            slot: self.slot,
            candidate_config_hash: &self.candidate_config_hash,
            coverage_receipt_sha256: &self.coverage_receipt_sha256,
        };
        hash_compact_json(READY_ID_DOMAIN_V1, &material)
    }

    pub fn request_identity_sha256(&self) -> &str {
        &self.request_identity_sha256
    }

    pub fn coverage_receipt_sha256(&self) -> &str {
        &self.coverage_receipt_sha256
    }

    pub(crate) fn sweep(&self) -> SweepId {
        self.sweep
    }

    pub fn ready_identity_sha256(&self) -> &str {
        &self.ready_identity_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuoteCoverageStateV1 {
    Awaiting(QuoteCoverageRequestV1),
    Ready {
        request: QuoteCoverageRequestV1,
        coverage: QuoteCoverageReadyV1,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteCoverageWaitReasonV1 {
    NoProvider,
    Pending,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwaitingQuoteCoverageV1 {
    request: QuoteCoverageRequestV1,
    reason: QuoteCoverageWaitReasonV1,
}

impl AwaitingQuoteCoverageV1 {
    pub fn new(request: QuoteCoverageRequestV1, reason: QuoteCoverageWaitReasonV1) -> Self {
        Self { request, reason }
    }

    pub fn request(&self) -> &QuoteCoverageRequestV1 {
        &self.request
    }

    pub fn reason(&self) -> QuoteCoverageWaitReasonV1 {
        self.reason
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuoteCoverageReadyBoundaryV1 {
    request: QuoteCoverageRequestV1,
    coverage: QuoteCoverageReadyV1,
}

impl QuoteCoverageReadyBoundaryV1 {
    pub fn new(
        request: QuoteCoverageRequestV1,
        coverage: QuoteCoverageReadyV1,
    ) -> Result<Self, QuoteCoverageErrorV1> {
        coverage.validate_against(&request)?;
        Ok(Self { request, coverage })
    }

    pub fn request(&self) -> &QuoteCoverageRequestV1 {
        &self.request
    }

    pub fn coverage(&self) -> &QuoteCoverageReadyV1 {
        &self.coverage
    }
}

#[derive(Debug, Clone)]
pub enum AutoresearchRunOutcomeV1 {
    Terminal(crate::verdict::SessionVerdict),
    AwaitingQuoteCoverage(AwaitingQuoteCoverageV1),
    QuoteCoverageReady(QuoteCoverageReadyBoundaryV1),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuoteCoverageProviderOutcomeV1 {
    Pending,
    Cancelled,
    Ready(QuoteCoverageReadyV1),
}

pub trait QuoteCoverageProviderV1 {
    fn provide_quote_coverage_v1(
        &mut self,
        request: &QuoteCoverageRequestV1,
    ) -> Result<QuoteCoverageProviderOutcomeV1, QuoteCoverageProviderErrorV1>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuoteCoverageProviderErrorV1 {
    detail: String,
}

impl QuoteCoverageProviderErrorV1 {
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl fmt::Display for QuoteCoverageProviderErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for QuoteCoverageProviderErrorV1 {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoresearchNonterminalBoundaryErrorV1 {
    boundary: &'static str,
    request_identity_sha256: String,
}

impl AutoresearchNonterminalBoundaryErrorV1 {
    pub(crate) fn awaiting(request: &QuoteCoverageRequestV1) -> Self {
        Self {
            boundary: "AwaitingQuoteCoverageV1",
            request_identity_sha256: request.request_identity_sha256.clone(),
        }
    }

    pub(crate) fn ready(request: &QuoteCoverageRequestV1) -> Self {
        Self {
            boundary: "QuoteCoverageReadyV1",
            request_identity_sha256: request.request_identity_sha256.clone(),
        }
    }
}

impl fmt::Display for AutoresearchNonterminalBoundaryErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "legacy terminal-only autoresearch API reached nonterminal {} for request {}; use run_until_boundary_v1",
            self.boundary, self.request_identity_sha256
        )
    }
}

impl std::error::Error for AutoresearchNonterminalBoundaryErrorV1 {}

fn validate_bounded_text(name: &str, value: &str) -> Result<(), QuoteCoverageErrorV1> {
    if value.is_empty() || value.len() > MAX_IDENTITY_TEXT_BYTES_V1 {
        return Err(QuoteCoverageErrorV1::new(
            QuoteCoverageErrorCodeV1::UnsupportedV1Shape,
            format!("{name} must contain 1..={MAX_IDENTITY_TEXT_BYTES_V1} bytes"),
        ));
    }
    Ok(())
}

fn validate_lowerhex_sha256(name: &str, value: &str) -> Result<(), QuoteCoverageErrorV1> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(QuoteCoverageErrorV1::new(
            QuoteCoverageErrorCodeV1::InvalidIdentity,
            format!("{name} must be exactly 64 lowercase hexadecimal bytes"),
        ));
    }
    Ok(())
}

fn compact_json_count<T: Serialize>(value: &T) -> Result<usize, QuoteCoverageErrorV1> {
    let mut writer = CountingWriter { bytes: 0 };
    serde_json::to_writer(&mut writer, value).map_err(|error| {
        QuoteCoverageErrorV1::new(
            QuoteCoverageErrorCodeV1::SerializationFailed,
            format!("serializing compact quote-coverage JSON: {error}"),
        )
    })?;
    Ok(writer.bytes)
}

fn hash_compact_json<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, QuoteCoverageErrorV1> {
    let mut writer = BoundedShaWriter::new(MAX_QUOTE_COVERAGE_REQUEST_BYTES_V1);
    writer.write_all(domain).map_err(map_writer_error)?;
    serde_json::to_writer(&mut writer, value).map_err(|error| {
        QuoteCoverageErrorV1::new(
            QuoteCoverageErrorCodeV1::SerializationFailed,
            format!("hashing compact quote-coverage JSON: {error}"),
        )
    })?;
    Ok(lowerhex(&writer.hasher.finalize()))
}

fn map_writer_error(error: io::Error) -> QuoteCoverageErrorV1 {
    QuoteCoverageErrorV1::new(
        QuoteCoverageErrorCodeV1::RequestTooLarge,
        format!("bounded quote-coverage writer rejected input: {error}"),
    )
}

struct CountingWriter {
    bytes: usize,
}

impl Write for CountingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(buffer.len())
            .ok_or_else(|| io::Error::other("quote-coverage byte-count overflow"))?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct BoundedShaWriter {
    bytes: usize,
    cap: usize,
    hasher: Sha256,
}

impl BoundedShaWriter {
    fn new(cap: usize) -> Self {
        Self {
            bytes: 0,
            cap,
            hasher: Sha256::new(),
        }
    }
}

impl Write for BoundedShaWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let attempted = self
            .bytes
            .checked_add(buffer.len())
            .ok_or_else(|| io::Error::other("quote-coverage byte-count overflow"))?;
        if attempted > self.cap {
            return Err(io::Error::other("quote-coverage request exceeds byte cap"));
        }
        self.hasher.update(buffer);
        self.bytes = attempted;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn lowerhex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}
