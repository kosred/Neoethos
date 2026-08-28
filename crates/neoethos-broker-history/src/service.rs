use crate::bootstrap_writer::{
    BROKER_TRENDBARS_PER_CHUNK_LIMIT, BrokerTrendbarStreamRequest, ctrader_inclusive_wire_to_ms,
    publish_broker_trendbar_chunks, publish_broker_trendbar_chunks_exact,
};
use crate::ctrader_data::{CTraderAuthenticatedHistoricalSession, CTraderChartHistoryRequest};
use crate::ctrader_historical_admission::{
    ActiveHistoricalFetch, HistoricalFetchCancelOutcome, HistoricalFetchQueueStartError,
    HistoricalFetchStartError, HistoricalRequestCancellation, HistoricalRequestCancelled,
    begin_process_historical_fetch_queued, cancel_process_historical_fetch,
    is_historical_request_cancelled, process_historical_fetch_status,
};
use crate::ctrader_live_auth::{
    CTraderEnvironment, CTraderLiveAuthBackend, CTraderTokenRefreshRequest,
    ProductionCTraderLiveAuthBackend,
};
use crate::ctrader_messages::CTraderCannotRouteRequestError;
use crate::secure_store::production_ctrader_token_store;
use anyhow::{Context, Result, anyhow, bail};
use neoethos_core::CanonicalTimeframe;
use neoethos_data::core::dataset_manifest::PublishResult;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvChunk, CanonicalVolumeChunk,
    SelectedDatasetGenerationV1,
};
use std::convert::Infallible;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

const CTRADER_MAX_TIMESTAMP_MS: i64 = 2_147_483_646_000;
const TOKEN_REFRESH_WINDOW_SECS: i64 = 120;
const MAX_HISTORICAL_LOGICAL_CHUNKS: usize = 20_000;
const MAX_HISTORICAL_PAGES: usize = 20_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerEnvironment {
    Demo,
    Live,
}

impl BrokerEnvironment {
    pub const fn endpoint_host(self) -> &'static str {
        match self {
            Self::Demo => "demo.ctraderapi.com",
            Self::Live => "live.ctraderapi.com",
        }
    }

    pub const fn from_canonical(environment: neoethos_data::CTraderEnvironment) -> Self {
        match environment {
            neoethos_data::CTraderEnvironment::Demo => Self::Demo,
            neoethos_data::CTraderEnvironment::Live => Self::Live,
        }
    }

    const fn transport(self) -> CTraderEnvironment {
        match self {
            Self::Demo => CTraderEnvironment::Demo,
            Self::Live => CTraderEnvironment::Live,
        }
    }

    const fn canonical(self) -> neoethos_data::CTraderEnvironment {
        match self {
            Self::Demo => neoethos_data::CTraderEnvironment::Demo,
            Self::Live => neoethos_data::CTraderEnvironment::Live,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoricalCredentials {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: BrokerEnvironment,
    pub account_id: i64,
}

/// Credential material for one exact broker-truth acquisition request.
///
/// This type intentionally has no `Debug` implementation. It stays private to
/// this crate and can only be constructed by the exact environment/account
/// loader below.
pub(crate) struct ExactProductionBrokerTruthCredentialsV2 {
    client_id: String,
    client_secret: String,
    access_token: String,
    environment: BrokerEnvironment,
    account_id: i64,
}

impl ExactProductionBrokerTruthCredentialsV2 {
    pub(crate) fn client_id(&self) -> &str {
        &self.client_id
    }

    pub(crate) fn client_secret(&self) -> &str {
        &self.client_secret
    }

    pub(crate) fn access_token(&self) -> &str {
        &self.access_token
    }

    pub(crate) const fn environment(&self) -> BrokerEnvironment {
        self.environment
    }

    pub(crate) const fn account_id(&self) -> i64 {
        self.account_id
    }
}

#[derive(Clone, Debug)]
pub enum HistoricalCaptureTarget {
    NewIdentity,
    SelectedGeneration(SelectedDatasetGenerationV1),
}

#[derive(Debug)]
pub enum BrokerHistoryConflict {
    IdentityMismatch { detail: String },
    DatasetRootOccupied { detail: String },
}

impl BrokerHistoryConflict {
    pub const fn response_code(&self) -> &'static str {
        match self {
            Self::IdentityMismatch { .. } => "BROKER_IDENTITY_MISMATCH",
            Self::DatasetRootOccupied { .. } => "BROKER_DATASET_ALREADY_EXISTS",
        }
    }
}

impl fmt::Display for BrokerHistoryConflict {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdentityMismatch { detail } => {
                write!(formatter, "broker dataset identity conflict: {detail}")
            }
            Self::DatasetRootOccupied { detail } => {
                write!(formatter, "broker dataset root is occupied: {detail}")
            }
        }
    }
}

impl std::error::Error for BrokerHistoryConflict {}

impl HistoricalCaptureTarget {
    fn selected(&self) -> Option<&SelectedDatasetGenerationV1> {
        match self {
            Self::NewIdentity => None,
            Self::SelectedGeneration(selected) => Some(selected),
        }
    }

    fn expected_generation_for<'a>(
        &'a self,
        resolved_identity: &CanonicalDatasetIdentity,
    ) -> Result<Option<&'a str>> {
        let Some(selected) = self.selected() else {
            return Ok(None);
        };
        if selected.identity() != resolved_identity {
            return Err(BrokerHistoryConflict::IdentityMismatch {
                detail: format!(
                    "selected broker identity {} does not match resolved live cTrader identity {}; refusing refresh before page one",
                    selected.identity().to_path_component(),
                    resolved_identity.to_path_component()
                ),
            }
            .into());
        }
        Ok(Some(selected.generation_id()))
    }

    fn validate_resolved_identity(
        &self,
        data_root: &Path,
        resolved_identity: &CanonicalDatasetIdentity,
    ) -> Result<()> {
        if self.selected().is_some() {
            self.expected_generation_for(resolved_identity)?;
            return Ok(());
        }

        let dataset_root = neoethos_data::core::dataset_manifest::canonical_dataset_root(
            data_root,
            resolved_identity,
        )
        .map_err(|error| {
            anyhow!(BrokerHistoryConflict::DatasetRootOccupied {
                detail: format!(
                    "cannot prove vacancy for {}: {error}",
                    resolved_identity.to_path_component()
                ),
            })
        })?;
        match std::fs::symlink_metadata(&dataset_root) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Ok(metadata) => {
                let kind = if metadata.file_type().is_symlink() {
                    "symlink or junction"
                } else if metadata.is_dir() {
                    "directory"
                } else {
                    "filesystem entry"
                };
                Err(BrokerHistoryConflict::DatasetRootOccupied {
                    detail: format!(
                        "canonical broker dataset {} already has a {kind}; an exact selected generation receipt is required before capture",
                        resolved_identity.to_path_component()
                    ),
                }
                .into())
            }
            Err(error) => Err(BrokerHistoryConflict::DatasetRootOccupied {
                detail: format!(
                    "canonical broker dataset {} metadata cannot prove vacancy ({:?}: {error}); refusing capture",
                    resolved_identity.to_path_component(),
                    error.kind()
                ),
            }
            .into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HistoricalCaptureRequest {
    pub symbol: String,
    pub timeframe: CanonicalTimeframe,
    pub from_ms: i64,
    pub to_ms: i64,
    pub data_root: PathBuf,
    pub target: HistoricalCaptureTarget,
}

#[derive(Clone, Debug)]
pub struct HistoricalDownloadOutcome {
    pub symbol: String,
    pub timeframe: CanonicalTimeframe,
    pub bar_count: usize,
    pub written_path: PathBuf,
    pub oldest_ms: i64,
    pub durable_commit_id: String,
    pub selected_generation: SelectedDatasetGenerationV1,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HistoricalBar {
    pub timestamp_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedHistoricalSymbol {
    pub environment: BrokerEnvironment,
    pub server: String,
    pub account_id: i64,
    pub symbol_id: i64,
    pub symbol_name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HistoricalPage {
    pub symbol_id: i64,
    pub timeframe: CanonicalTimeframe,
    pub bars: Vec<HistoricalBar>,
    pub has_more: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HistoricalPageRequest {
    pub timeframe: CanonicalTimeframe,
    pub from_timestamp_ms: i64,
    pub to_timestamp_ms: i64,
    pub count: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AuthenticatedHistoricalSessionRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: BrokerEnvironment,
    pub server: String,
    pub account_id: i64,
    pub symbol_name: String,
    pub timeframe: CanonicalTimeframe,
    pub from_timestamp_ms: i64,
    pub to_timestamp_ms: i64,
}

pub(crate) trait HistoricalSession {
    fn resolved_symbol(&self) -> &ResolvedHistoricalSymbol;
    fn next_page(&mut self, request: HistoricalPageRequest) -> Result<HistoricalPage>;
}

pub(crate) trait HistoricalSessionConnector {
    type Session: HistoricalSession;

    fn connect_authenticated(
        &self,
        request: &AuthenticatedHistoricalSessionRequest,
        cancellation: &HistoricalRequestCancellation,
    ) -> Result<Self::Session>;
}

#[cfg(test)]
pub(crate) type ProductionPublication = PublishResult;

pub(crate) struct ProductionHistoricalSession {
    inner: CTraderAuthenticatedHistoricalSession,
    resolved: ResolvedHistoricalSymbol,
}

impl HistoricalSession for ProductionHistoricalSession {
    fn resolved_symbol(&self) -> &ResolvedHistoricalSymbol {
        &self.resolved
    }

    fn next_page(&mut self, request: HistoricalPageRequest) -> Result<HistoricalPage> {
        let page = self.inner.next_trendbars(
            request.timeframe,
            request.from_timestamp_ms,
            request.to_timestamp_ms,
            request.count,
        )?;
        page.validate_identity(self.resolved.symbol_id, request.timeframe)?;
        Ok(HistoricalPage {
            symbol_id: page.symbol_id,
            timeframe: page.timeframe,
            bars: page
                .bars
                .into_iter()
                .map(|bar| HistoricalBar {
                    timestamp_ms: bar.timestamp_ms,
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                    volume: bar.volume,
                })
                .collect(),
            has_more: page.has_more,
        })
    }
}

pub(crate) struct ProductionHistoricalSessionConnector;

const CANNOT_ROUTE_CONNECT_ATTEMPTS: usize = 5;

pub(crate) fn connect_historical_session_with_route_retry<T>(
    cancellation: &HistoricalRequestCancellation,
    mut connect: impl FnMut() -> Result<T>,
    mut wait: impl FnMut(Duration) -> Result<()>,
) -> Result<T> {
    for attempt in 1..=CANNOT_ROUTE_CONNECT_ATTEMPTS {
        ensure_not_cancelled(cancellation)?;
        match connect() {
            Ok(session) => return Ok(session),
            Err(error) => {
                let cannot_route = error
                    .downcast_ref::<CTraderCannotRouteRequestError>()
                    .is_some();
                if !cannot_route || attempt == CANNOT_ROUTE_CONNECT_ATTEMPTS {
                    return Err(error);
                }
                let delay_seconds = 1_u64 << (attempt - 1);
                wait(Duration::from_secs(delay_seconds))?;
            }
        }
    }
    unreachable!("bounded cTrader route retry loop returns on every terminal branch")
}

impl HistoricalSessionConnector for ProductionHistoricalSessionConnector {
    type Session = ProductionHistoricalSession;

    fn connect_authenticated(
        &self,
        request: &AuthenticatedHistoricalSessionRequest,
        cancellation: &HistoricalRequestCancellation,
    ) -> Result<Self::Session> {
        let transport_request = CTraderChartHistoryRequest {
            client_id: request.client_id.clone(),
            client_secret: request.client_secret.clone(),
            access_token: request.access_token.clone(),
            environment: request.environment.transport(),
            account_id: request.account_id.to_string(),
            symbol_name: request.symbol_name.clone(),
            timeframe: request.timeframe.to_string(),
            from_timestamp_ms: request.from_timestamp_ms,
            to_timestamp_ms: request.to_timestamp_ms,
            count: None,
        };
        let inner = connect_historical_session_with_route_retry(
            cancellation,
            || CTraderAuthenticatedHistoricalSession::connect(&transport_request, cancellation),
            |delay| {
                if cancellation.wait_for_cancellation(delay) {
                    Err(anyhow!(HistoricalRequestCancelled))
                } else {
                    Ok(())
                }
            },
        )?;
        let symbol = &inner.resolved_symbol().symbol;
        let resolved = ResolvedHistoricalSymbol {
            environment: request.environment,
            server: request.server.clone(),
            account_id: request.account_id,
            symbol_id: symbol.symbol_id,
            symbol_name: symbol.symbol_name.clone(),
        };
        Ok(ProductionHistoricalSession { inner, resolved })
    }
}

fn ensure_not_cancelled(cancellation: &HistoricalRequestCancellation) -> Result<()> {
    if cancellation.is_cancelled() {
        return Err(anyhow!(HistoricalRequestCancelled));
    }
    Ok(())
}

fn normalize_historical_request_error(error: anyhow::Error) -> anyhow::Error {
    if is_historical_request_cancelled(error.as_ref()) {
        anyhow!(HistoricalRequestCancelled)
    } else {
        error
    }
}

pub fn is_historical_capture_cancelled(error: &(dyn std::error::Error + 'static)) -> bool {
    is_historical_request_cancelled(error)
}

fn timeframe_chunk_ms(timeframe: CanonicalTimeframe) -> Result<Option<i64>> {
    if timeframe == CanonicalTimeframe::D1 {
        return Ok(Some(4_000 * 86_400_000));
    }
    timeframe
        .fixed_duration_ms()
        .map(|duration| {
            duration
                .checked_mul(4_320)
                .context("cTrader timeframe chunk width overflows i64 milliseconds")
        })
        .transpose()
}

fn broker_bars_are_bit_identical(left: &HistoricalBar, right: &HistoricalBar) -> bool {
    left.timestamp_ms == right.timestamp_ms
        && left.open.to_bits() == right.open.to_bits()
        && left.high.to_bits() == right.high.to_bits()
        && left.low.to_bits() == right.low.to_bits()
        && left.close.to_bits() == right.close.to_bits()
        && left.volume == right.volume
}

fn normalize_broker_bar_order(
    bars: Vec<HistoricalBar>,
    context: &str,
) -> Result<Vec<HistoricalBar>> {
    let mut normalized = Vec::with_capacity(bars.len());
    for (row, bar) in bars.into_iter().enumerate() {
        let Some(previous) = normalized.last() else {
            normalized.push(bar);
            continue;
        };
        if bar.timestamp_ms > previous.timestamp_ms {
            normalized.push(bar);
            continue;
        }
        if bar.timestamp_ms == previous.timestamp_ms
            && broker_bars_are_bit_identical(previous, &bar)
        {
            continue;
        }
        if bar.timestamp_ms == previous.timestamp_ms {
            bail!(
                "{context} contains the same timestamp with different OHLCV at rows {}/{}: left={previous:?}; right={bar:?}; refusing ambiguous broker data",
                row - 1,
                row
            );
        }
        bail!(
            "{context} descends at rows {}/{}: {} -> {}; left={previous:?}; right={bar:?}; refusing sort repair",
            row - 1,
            row,
            previous.timestamp_ms,
            bar.timestamp_ms
        );
    }
    Ok(normalized)
}

fn historical_bars_into_chunk(
    bars: Vec<HistoricalBar>,
    requested_from_ms: i64,
    requested_to_ms: i64,
    symbol: &str,
    timeframe: CanonicalTimeframe,
) -> Result<CanonicalOhlcvChunk> {
    if bars.is_empty() {
        bail!("cannot spool an empty cTrader history page");
    }
    let leading_outside = bars
        .iter()
        .take_while(|bar| bar.timestamp_ms < requested_from_ms)
        .count();
    if leading_outside > 0 {
        let maximum_overlap_ms = timeframe
            .fixed_duration_ms()
            .unwrap_or_else(|| match timeframe {
                CanonicalTimeframe::D1 => 24 * 60 * 60 * 1_000,
                CanonicalTimeframe::W1 => 7 * 24 * 60 * 60 * 1_000,
                CanonicalTimeframe::MN1 => 31 * 24 * 60 * 60 * 1_000,
                _ => unreachable!("every canonical timeframe has a containing-bar bound"),
            });
        if leading_outside != 1 {
            bail!(
                "cTrader {symbol} {timeframe} returned {leading_outside} trendbars before the half-open request lower bound; expected at most one containing bar"
            );
        }
        let leading_timestamp_ms = bars[0].timestamp_ms;
        let overlap_age_ms = requested_from_ms - leading_timestamp_ms;
        if overlap_age_ms <= 0 || overlap_age_ms > maximum_overlap_ms {
            bail!(
                "cTrader {symbol} {timeframe} leading containing bar at {leading_timestamp_ms} is too far before the requested lower bound {requested_from_ms}"
            );
        }
    }
    let in_range_len = bars.len() - leading_outside;
    if in_range_len == 0 {
        bail!(
            "cTrader {symbol} {timeframe} page contains no bar whose canonical open is inside [{requested_from_ms}, {requested_to_ms})"
        );
    }
    let carries_volume = bars[leading_outside].volume.is_some();
    let mut timestamp_ms = Vec::with_capacity(in_range_len);
    let mut open = Vec::with_capacity(in_range_len);
    let mut high = Vec::with_capacity(in_range_len);
    let mut low = Vec::with_capacity(in_range_len);
    let mut close = Vec::with_capacity(in_range_len);
    let mut volume = carries_volume.then(|| Vec::with_capacity(in_range_len));
    for (row, bar) in bars.into_iter().skip(leading_outside).enumerate() {
        if bar.timestamp_ms >= requested_to_ms {
            let source_row = row + leading_outside;
            bail!(
                "cTrader {symbol} {timeframe} trendbar row {source_row} timestamp {} is outside the half-open request [{requested_from_ms}, {requested_to_ms})",
                bar.timestamp_ms
            );
        }
        if [bar.open, bar.high, bar.low, bar.close]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!(
                "cTrader returned a non-finite or non-positive {symbol} {timeframe} OHLC row at {}",
                bar.timestamp_ms
            );
        }
        match (&mut volume, bar.volume) {
            (Some(values), Some(value)) if value >= 0 => values.push(value),
            (None, None) => {}
            (Some(_), Some(value)) => bail!(
                "cTrader returned negative {symbol} {timeframe} tick volume {value} at {}",
                bar.timestamp_ms
            ),
            _ => bail!(
                "cTrader tick volume presence changed inside {symbol} {timeframe} page at row {row}"
            ),
        }
        timestamp_ms.push(bar.timestamp_ms);
        open.push(bar.open);
        high.push(bar.high);
        low.push(bar.low);
        close.push(bar.close);
    }
    Ok(CanonicalOhlcvChunk {
        timestamp_ms,
        open,
        high,
        low,
        close,
        volume: volume.map_or(CanonicalVolumeChunk::Absent, CanonicalVolumeChunk::Int64),
    })
}

fn publish_history<I>(
    request: &HistoricalCaptureRequest,
    identity: &CanonicalDatasetIdentity,
    retrieved_unix_ms: u64,
    returned_from_ms: i64,
    returned_to_ms: i64,
    row_count: u64,
    chunks: I,
) -> Result<PublishResult>
where
    I: IntoIterator<Item = Result<CanonicalOhlcvChunk>>,
{
    let expected_generation = request.target.expected_generation_for(identity)?;
    let publication_request = BrokerTrendbarStreamRequest {
        configured_root: &request.data_root,
        identity,
        expected_generation,
        requested_from_ms: request.from_ms,
        requested_to_ms: request.to_ms,
        retrieved_unix_ms,
        returned_from_ms,
        returned_to_ms,
        row_count,
        chunks,
    };
    match &request.target {
        HistoricalCaptureTarget::NewIdentity => publish_broker_trendbar_chunks(publication_request),
        HistoricalCaptureTarget::SelectedGeneration(selected) => {
            publish_broker_trendbar_chunks_exact(publication_request, selected)
        }
    }
}

fn authenticated_session_request_for_capture(
    request: &HistoricalCaptureRequest,
    credentials: &HistoricalCredentials,
) -> Result<AuthenticatedHistoricalSessionRequest> {
    if request.to_ms <= request.from_ms {
        bail!(
            "invalid range: from_ms ({}) must be < to_ms ({})",
            request.from_ms,
            request.to_ms
        );
    }
    if request.from_ms < 0 || request.to_ms > CTRADER_MAX_TIMESTAMP_MS {
        bail!("cTrader trendbar range must be within 0..={CTRADER_MAX_TIMESTAMP_MS} ms");
    }
    let symbol_assertion = request.symbol.trim();
    if symbol_assertion.is_empty() {
        bail!("cTrader historical symbol must be non-empty");
    }
    if let Some(selected) = request.target.selected() {
        if !selected
            .identity()
            .symbol_name()
            .eq_ignore_ascii_case(symbol_assertion)
        {
            return Err(BrokerHistoryConflict::IdentityMismatch {
                detail: format!(
                    "fetch symbol assertion {:?} does not match selected identity symbol {:?}",
                    request.symbol,
                    selected.identity().symbol_name()
                ),
            }
            .into());
        }
        if selected.identity().timeframe() != request.timeframe {
            return Err(BrokerHistoryConflict::IdentityMismatch {
                detail: format!(
                    "fetch timeframe assertion {} does not match selected identity timeframe {}",
                    request.timeframe,
                    selected.identity().timeframe()
                ),
            }
            .into());
        }
        let _ = neoethos_data::open_exact_dataset_generation(&request.data_root, selected)?;
    }
    let requested_symbol = request
        .target
        .selected()
        .map_or(symbol_assertion, |selected| {
            selected.identity().symbol_name()
        });
    Ok(AuthenticatedHistoricalSessionRequest {
        client_id: credentials.client_id.clone(),
        client_secret: credentials.client_secret.clone(),
        access_token: credentials.access_token.clone(),
        environment: credentials.environment,
        server: credentials.environment.endpoint_host().to_owned(),
        account_id: credentials.account_id,
        symbol_name: requested_symbol.to_owned(),
        timeframe: request.timeframe,
        from_timestamp_ms: request.from_ms,
        to_timestamp_ms: ctrader_inclusive_wire_to_ms(request.to_ms)?,
    })
}

pub(crate) struct HistoricalSeriesCapture<C: HistoricalSessionConnector> {
    credentials: HistoricalCredentials,
    connector: C,
    session: Option<C::Session>,
}

impl<C: HistoricalSessionConnector> HistoricalSeriesCapture<C> {
    pub(crate) fn new(credentials: HistoricalCredentials, connector: C) -> Self {
        Self {
            credentials,
            connector,
            session: None,
        }
    }

    pub(crate) fn capture_with_publication_hook<H>(
        &mut self,
        request: HistoricalCaptureRequest,
        active_fetch: &ActiveHistoricalFetch<'_>,
        after_publication: H,
    ) -> Result<HistoricalDownloadOutcome>
    where
        H: FnOnce(&PublishResult) -> Result<()>,
    {
        let cancellation = active_fetch.cancellation();
        ensure_not_cancelled(cancellation)?;
        let session_request =
            authenticated_session_request_for_capture(&request, &self.credentials)?;
        let reuse = self.session.as_ref().is_some_and(|session| {
            let resolved = session.resolved_symbol();
            resolved.environment == session_request.environment
                && resolved.server == session_request.server
                && resolved.account_id == session_request.account_id
                && resolved
                    .symbol_name
                    .eq_ignore_ascii_case(&session_request.symbol_name)
        });
        if !reuse {
            self.session = Some(
                self.connector
                    .connect_authenticated(&session_request, cancellation)
                    .map_err(normalize_historical_request_error)?,
            );
        }
        let session = self
            .session
            .as_mut()
            .context("authenticated historical series session is unavailable")?;
        capture_with_session_and_publication_hook(request, active_fetch, session, after_publication)
    }

    pub(crate) fn capture_historical_series_generation(
        &mut self,
        request: HistoricalCaptureRequest,
        active_fetch: &ProcessHistoricalCapture,
    ) -> Result<HistoricalDownloadOutcome> {
        active_fetch.active.execute_if_not_cancelled(|active| {
            self.capture_with_publication_hook(request, active, |_| Ok(()))
        })?
    }
}

pub(crate) fn capture_with_connector_and_publication_hook<C, H>(
    request: HistoricalCaptureRequest,
    credentials: HistoricalCredentials,
    active_fetch: &ActiveHistoricalFetch<'_>,
    connector: &C,
    after_publication: H,
) -> Result<HistoricalDownloadOutcome>
where
    C: HistoricalSessionConnector,
    H: FnOnce(&PublishResult) -> Result<()>,
{
    let cancellation = active_fetch.cancellation();
    ensure_not_cancelled(cancellation)?;
    let session_request = authenticated_session_request_for_capture(&request, &credentials)?;
    let mut session = connector
        .connect_authenticated(&session_request, cancellation)
        .map_err(normalize_historical_request_error)?;
    capture_with_session_and_publication_hook(
        request,
        active_fetch,
        &mut session,
        after_publication,
    )
}

fn capture_with_session_and_publication_hook<S, H>(
    request: HistoricalCaptureRequest,
    active_fetch: &ActiveHistoricalFetch<'_>,
    session: &mut S,
    after_publication: H,
) -> Result<HistoricalDownloadOutcome>
where
    S: HistoricalSession,
    H: FnOnce(&PublishResult) -> Result<()>,
{
    let cancellation = active_fetch.cancellation();
    ensure_not_cancelled(cancellation)?;
    if request.to_ms <= request.from_ms {
        bail!(
            "invalid range: from_ms ({}) must be < to_ms ({})",
            request.from_ms,
            request.to_ms
        );
    }
    if request.from_ms < 0 || request.to_ms > CTRADER_MAX_TIMESTAMP_MS {
        bail!("cTrader trendbar range must be within 0..={CTRADER_MAX_TIMESTAMP_MS} ms");
    }
    let symbol_assertion = request.symbol.trim();
    if symbol_assertion.is_empty() {
        bail!("cTrader historical symbol must be non-empty");
    }

    let _selected_generation_lease = match request.target.selected() {
        Some(selected) => {
            if !selected
                .identity()
                .symbol_name()
                .eq_ignore_ascii_case(symbol_assertion)
            {
                return Err(BrokerHistoryConflict::IdentityMismatch {
                    detail: format!(
                        "fetch symbol assertion {:?} does not match selected identity symbol {:?}",
                        request.symbol,
                        selected.identity().symbol_name()
                    ),
                }
                .into());
            }
            if selected.identity().timeframe() != request.timeframe {
                return Err(BrokerHistoryConflict::IdentityMismatch {
                    detail: format!(
                        "fetch timeframe assertion {} does not match selected identity timeframe {}",
                        request.timeframe,
                        selected.identity().timeframe()
                    ),
                }
                .into());
            }
            let (_, lease) =
                neoethos_data::open_exact_dataset_generation(&request.data_root, selected)?;
            Some(lease)
        }
        None => None,
    };
    let requested_symbol = request
        .target
        .selected()
        .map_or(symbol_assertion, |selected| {
            selected.identity().symbol_name()
        });
    let resolved = session.resolved_symbol().clone();
    if !resolved.symbol_name.eq_ignore_ascii_case(requested_symbol) {
        bail!(
            "authenticated historical series session is bound to symbol {:?}, not requested symbol {:?}",
            resolved.symbol_name,
            requested_symbol
        );
    }
    let identity = CanonicalDatasetIdentity::ctrader(
        resolved.environment.canonical(),
        &resolved.server,
        resolved.account_id,
        resolved.symbol_id,
        &resolved.symbol_name,
        request.timeframe,
        BarTimestampConvention::BarOpen,
    )
    .map_err(|error| anyhow!(error.to_string()))?;
    request
        .target
        .validate_resolved_identity(&request.data_root, &identity)?;

    let chunk_ms = timeframe_chunk_ms(request.timeframe)?;
    let span_ms = request.to_ms - request.from_ms;
    let needed_chunks = chunk_ms.map_or(1, |width| {
        span_ms
            .saturating_add(width)
            .saturating_sub(1)
            .checked_div(width)
            .unwrap_or(0)
            .saturating_add(2)
    });
    let max_chunks = needed_chunks.clamp(1, MAX_HISTORICAL_LOGICAL_CHUNKS as i64) as usize;
    let mut cursor_to = request.to_ms;
    let mut spool = None;
    let mut chunk_count = 0_usize;
    let mut page_count = 0_usize;
    let mut newer_chunk_first_ms = None;
    let mut returned_from_ms = None;
    let mut returned_to_ms = None;
    let mut row_count = 0_u64;
    while cursor_to > request.from_ms && chunk_count < max_chunks {
        let page_to_ms = cursor_to;
        let cursor_from = chunk_ms
            .map(|width| page_to_ms.saturating_sub(width).max(request.from_ms))
            .unwrap_or(request.from_ms);
        chunk_count += 1;
        let mut subpage_to_ms = page_to_ms;
        loop {
            if page_count >= MAX_HISTORICAL_PAGES {
                bail!(
                    "cTrader history traversal exceeded its hard page limit of {MAX_HISTORICAL_PAGES} before reaching a terminal response"
                );
            }
            let mut page = session
                .next_page(HistoricalPageRequest {
                    timeframe: request.timeframe,
                    from_timestamp_ms: cursor_from,
                    to_timestamp_ms: ctrader_inclusive_wire_to_ms(subpage_to_ms)?,
                    count: None,
                })
                .map_err(normalize_historical_request_error)?;
            page_count += 1;
            ensure_not_cancelled(cancellation)?;
            if page.symbol_id != resolved.symbol_id || page.timeframe != request.timeframe {
                bail!("cTrader page identity changed inside the persistent historical session");
            }
            if page.bars.is_empty() {
                if page.has_more {
                    bail!(
                        "cTrader reported hasMore with an empty {} {} page for [{cursor_from}, {subpage_to_ms}); refusing a non-progressing traversal",
                        resolved.symbol_name,
                        request.timeframe
                    );
                }
                break;
            }
            if page.bars.len() > BROKER_TRENDBARS_PER_CHUNK_LIMIT {
                bail!(
                    "cTrader returned {} {} {} trendbars in one page, above the bounded-page limit of {BROKER_TRENDBARS_PER_CHUNK_LIMIT}",
                    page.bars.len(),
                    resolved.symbol_name,
                    request.timeframe
                );
            }
            page.bars = normalize_broker_bar_order(page.bars, "cTrader history page")?;
            let page_first_ms = page.bars[0].timestamp_ms;
            let has_more = page.has_more;
            let chunk = historical_bars_into_chunk(
                page.bars,
                cursor_from,
                subpage_to_ms,
                &resolved.symbol_name,
                request.timeframe,
            )?;
            let chunk_first_ms = *chunk
                .timestamp_ms
                .first()
                .context("validated cTrader page has no in-range first bar")?;
            let chunk_last_ms = *chunk
                .timestamp_ms
                .last()
                .context("validated cTrader page has no in-range last bar")?;
            if newer_chunk_first_ms.is_some_and(|newer| chunk_last_ms >= newer) {
                bail!(
                    "cTrader validated half-open history chunks overlap or descend at {chunk_last_ms}; refusing sort/dedup repair"
                );
            }
            let page_rows = u64::try_from(chunk.timestamp_ms.len()).context("page row count")?;
            row_count = row_count
                .checked_add(page_rows)
                .context("history row count")?;
            returned_to_ms.get_or_insert(chunk_last_ms);
            returned_from_ms = Some(chunk_first_ms);
            newer_chunk_first_ms = Some(chunk_first_ms);
            if spool.is_none() {
                spool = Some(neoethos_data::CanonicalOhlcvReverseSpool::create(
                    &request.data_root,
                    BROKER_TRENDBARS_PER_CHUNK_LIMIT,
                    MAX_HISTORICAL_PAGES,
                )?);
            }
            spool
                .as_mut()
                .context("cTrader Vortex spool was not initialized")?
                .push_latest(chunk)?;

            if !has_more {
                break;
            }
            if page_first_ms <= cursor_from {
                bail!(
                    "cTrader reported hasMore without a strictly older {} {} cursor inside [{cursor_from}, {page_to_ms}); refusing a non-progressing traversal",
                    resolved.symbol_name,
                    request.timeframe
                );
            }
            subpage_to_ms = page_first_ms;
        }
        cursor_to = cursor_from;
    }
    if cursor_to > request.from_ms {
        bail!(
            "cTrader history traversal stopped at {cursor_to} before requested lower bound {}",
            request.from_ms
        );
    }
    if row_count == 0 {
        bail!(
            "cTrader returned no {} {} bars for [{}, {}); refusing an empty generation",
            resolved.symbol_name,
            request.timeframe,
            request.from_ms,
            request.to_ms
        );
    }
    let returned_from_ms = returned_from_ms.context("cTrader stream has no oldest bar")?;
    let returned_to_ms = returned_to_ms.context("cTrader stream has no newest bar")?;
    let spool = spool.context("cTrader stream has no validated Vortex pages")?;
    let retrieved_unix_ms = u64::try_from(chrono::Utc::now().timestamp_millis())
        .context("current timestamp is before Unix epoch")?;
    ensure_not_cancelled(cancellation)?;
    let publication_permit = active_fetch.begin_publication()?;
    debug_assert_eq!(publication_permit.run_id(), active_fetch.run_id());
    let publication = publish_history(
        &request,
        &identity,
        retrieved_unix_ms,
        returned_from_ms,
        returned_to_ms,
        row_count,
        spool.into_oldest_first(),
    )?;
    after_publication(&publication)?;
    let selected_generation = SelectedDatasetGenerationV1::from_manifest(publication.manifest())?;
    Ok(HistoricalDownloadOutcome {
        symbol: identity.symbol_name().to_owned(),
        timeframe: request.timeframe,
        bar_count: usize::try_from(row_count).context("broker row count exceeds usize")?,
        written_path: publication.manifest().generation_path(),
        oldest_ms: returned_from_ms,
        durable_commit_id: publication.durable_commit_id().to_owned(),
        selected_generation,
    })
}

pub fn capture_historical_generation(
    request: HistoricalCaptureRequest,
    credentials: HistoricalCredentials,
    active_fetch: &ProcessHistoricalCapture,
) -> Result<HistoricalDownloadOutcome> {
    active_fetch.active.execute_if_not_cancelled(|active| {
        capture_with_connector_and_publication_hook(
            request,
            credentials,
            active,
            &ProductionHistoricalSessionConnector,
            |_| Ok(()),
        )
    })?
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn load_fresh_token(client_id: &str, client_secret: &str) -> Result<String> {
    let store = production_ctrader_token_store();
    let bundle = store
        .load_token_bundle_with_legacy_fallback()
        .context("load cTrader token bundle")?
        .context("no cTrader token bundle is stored; authenticate once before historical fetch")?;
    if !bundle.needs_refresh_at(now_unix(), TOKEN_REFRESH_WINDOW_SECS) {
        return Ok(bundle.access_token);
    }
    if bundle.refresh_token.trim().is_empty() {
        bail!("stored cTrader token is expired and has no refresh token");
    }
    let refresh = CTraderTokenRefreshRequest {
        client_id: client_id.to_owned(),
        client_secret: client_secret.to_owned(),
        refresh_token: bundle.refresh_token,
        scope: if bundle.scope.trim().is_empty() {
            "trading".to_owned()
        } else {
            bundle.scope
        },
    };
    let fresh = ProductionCTraderLiveAuthBackend
        .refresh_token_bundle(&refresh)
        .context("refresh cTrader access token")?;
    store
        .save_token_bundle(&fresh)
        .context("persist refreshed cTrader token")?;
    Ok(fresh.access_token)
}

fn load_exact_fresh_token(client_id: &str, client_secret: &str) -> Result<String> {
    let store = production_ctrader_token_store();
    let bundle = store
        .load_token_bundle()
        .context("load exact cTrader token bundle")?
        .context("no exact cTrader token bundle is stored")?;
    if !bundle.needs_refresh_at(now_unix(), TOKEN_REFRESH_WINDOW_SECS) {
        return Ok(bundle.access_token);
    }
    if bundle.refresh_token.trim().is_empty() {
        bail!("exact stored cTrader token is expired and has no refresh token");
    }
    if bundle.scope.trim().is_empty() {
        bail!("exact stored cTrader token is expired and has no refresh scope");
    }
    let refresh = CTraderTokenRefreshRequest {
        client_id: client_id.to_owned(),
        client_secret: client_secret.to_owned(),
        refresh_token: bundle.refresh_token,
        scope: bundle.scope,
    };
    let fresh = ProductionCTraderLiveAuthBackend
        .refresh_token_bundle(&refresh)
        .context("refresh exact cTrader access token")?;
    store
        .save_token_bundle(&fresh)
        .context("persist refreshed exact cTrader token")?;
    Ok(fresh.access_token)
}

pub(crate) fn load_exact_production_broker_truth_credentials_v2(
    expected_environment: BrokerEnvironment,
    expected_account_id: i64,
) -> Result<ExactProductionBrokerTruthCredentialsV2> {
    if expected_account_id <= 0 {
        bail!("exact cTrader account id must be positive");
    }
    let path = neoethos_core::broker_config::credentials_file_path()
        .context("resolve broker credentials path")?;
    let settings = neoethos_core::broker_config::load_from_disk(&path)
        .with_context(|| format!("load broker credentials from {}", path.display()))?
        .with_context(|| format!("broker credentials file {} does not exist", path.display()))?;
    let ctrader = settings.ctrader;
    if ctrader.client_id.trim().is_empty() || ctrader.client_secret.trim().is_empty() {
        bail!("cTrader client_id/client_secret are missing from broker credentials");
    }
    let configured_environment = match ctrader.environment {
        neoethos_core::broker_config::CTraderBrokerEnvironment::Demo => BrokerEnvironment::Demo,
        neoethos_core::broker_config::CTraderBrokerEnvironment::Live => BrokerEnvironment::Live,
    };
    if configured_environment != expected_environment {
        bail!("configured cTrader environment does not match the exact acquisition request");
    }

    let mut exact_account_matches = 0_usize;
    for account in &ctrader.accounts {
        let configured_account_id = account
            .account_id
            .trim()
            .parse::<i64>()
            .context("configured cTrader account id is not numeric")?;
        if configured_account_id == expected_account_id {
            exact_account_matches = exact_account_matches
                .checked_add(1)
                .context("configured cTrader account match count overflow")?;
        }
    }
    if exact_account_matches != 1 {
        bail!("exact cTrader account is missing or configured more than once");
    }

    let access_token = load_exact_fresh_token(&ctrader.client_id, &ctrader.client_secret)?;
    Ok(ExactProductionBrokerTruthCredentialsV2 {
        client_id: ctrader.client_id,
        client_secret: ctrader.client_secret,
        access_token,
        environment: configured_environment,
        account_id: expected_account_id,
    })
}

pub fn load_production_historical_credentials() -> Result<HistoricalCredentials> {
    let path = neoethos_core::broker_config::credentials_file_path()
        .context("resolve broker credentials path")?;
    let settings = neoethos_core::broker_config::load_from_disk(&path)
        .with_context(|| format!("load broker credentials from {}", path.display()))?
        .with_context(|| format!("broker credentials file {} does not exist", path.display()))?;
    let ctrader = settings.ctrader;
    if ctrader.client_id.trim().is_empty() || ctrader.client_secret.trim().is_empty() {
        bail!("cTrader client_id/client_secret are missing from broker credentials");
    }
    let account = ctrader
        .accounts
        .iter()
        .find(|account| account.enabled_for_execution)
        .or_else(|| ctrader.accounts.first())
        .context("no cTrader account is configured")?;
    let account_id = account
        .account_id
        .parse::<i64>()
        .context("configured cTrader account id is not numeric")?;
    let access_token = load_fresh_token(&ctrader.client_id, &ctrader.client_secret)?;
    let environment = match ctrader.environment {
        neoethos_core::broker_config::CTraderBrokerEnvironment::Demo => BrokerEnvironment::Demo,
        neoethos_core::broker_config::CTraderBrokerEnvironment::Live => BrokerEnvironment::Live,
    };
    Ok(HistoricalCredentials {
        client_id: ctrader.client_id,
        client_secret: ctrader.client_secret,
        access_token,
        environment,
        account_id,
    })
}

pub fn load_exact_production_historical_credentials(
    expected_environment: BrokerEnvironment,
    expected_account_id: i64,
) -> Result<HistoricalCredentials> {
    if expected_account_id <= 0 {
        bail!("exact cTrader historical account id must be positive");
    }
    let path = neoethos_core::broker_config::credentials_file_path()
        .context("resolve broker credentials path")?;
    let settings = neoethos_core::broker_config::load_from_disk(&path)
        .with_context(|| format!("load broker credentials from {}", path.display()))?
        .with_context(|| format!("broker credentials file {} does not exist", path.display()))?;
    let ctrader = settings.ctrader;
    if ctrader.client_id.trim().is_empty() || ctrader.client_secret.trim().is_empty() {
        bail!("cTrader client_id/client_secret are missing from broker credentials");
    }
    let configured_environment = match ctrader.environment {
        neoethos_core::broker_config::CTraderBrokerEnvironment::Demo => BrokerEnvironment::Demo,
        neoethos_core::broker_config::CTraderBrokerEnvironment::Live => BrokerEnvironment::Live,
    };
    if configured_environment != expected_environment {
        bail!("configured cTrader environment does not match the exact historical plan");
    }
    let mut exact_matches = 0_usize;
    for account in &ctrader.accounts {
        let account_id = account
            .account_id
            .trim()
            .parse::<i64>()
            .context("configured cTrader account id is not numeric")?;
        if account_id == expected_account_id {
            exact_matches = exact_matches
                .checked_add(1)
                .context("exact cTrader historical account match count overflow")?;
        }
    }
    if exact_matches != 1 {
        bail!("exact cTrader historical account is missing or configured more than once");
    }
    let access_token = load_exact_fresh_token(&ctrader.client_id, &ctrader.client_secret)?;
    Ok(HistoricalCredentials {
        client_id: ctrader.client_id,
        client_secret: ctrader.client_secret,
        access_token,
        environment: configured_environment,
        account_id: expected_account_id,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoricalFetchCancelResult {
    Cancelled {
        run_id: u64,
    },
    PublicationInProgress {
        run_id: u64,
    },
    StaleRun {
        requested_run_id: u64,
        active_run_id: u64,
    },
    NoActiveFetch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoricalFetchStartFailure {
    AlreadyActive { active_run_id: u64 },
    RunIdOverflow,
    Cancelled { run_id: u64 },
}

impl fmt::Display for HistoricalFetchStartFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyActive { active_run_id } => {
                write!(
                    formatter,
                    "historical fetch {active_run_id} is already active"
                )
            }
            Self::RunIdOverflow => formatter.write_str("historical fetch run id space exhausted"),
            Self::Cancelled { run_id } => {
                write!(
                    formatter,
                    "historical fetch {run_id} was cancelled before admission"
                )
            }
        }
    }
}

impl std::error::Error for HistoricalFetchStartFailure {}

pub struct ProcessHistoricalCapture {
    active: ActiveHistoricalFetch<'static>,
}

impl ProcessHistoricalCapture {
    pub fn run_id(&self) -> u64 {
        self.active.run_id()
    }

    pub fn cancellation_handle(&self) -> HistoricalCaptureCancellationHandle {
        HistoricalCaptureCancellationHandle {
            run_id: self.run_id(),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.active.cancellation().is_cancelled()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoricalCaptureCancellationHandle {
    run_id: u64,
}

impl HistoricalCaptureCancellationHandle {
    pub const fn run_id(self) -> u64 {
        self.run_id
    }

    pub fn cancel(self) -> HistoricalFetchCancelResult {
        cancel_process_historical_capture(self.run_id)
    }
}

pub fn cancel_process_historical_capture(run_id: u64) -> HistoricalFetchCancelResult {
    match cancel_process_historical_fetch(run_id) {
        HistoricalFetchCancelOutcome::Cancelled { run_id } => {
            HistoricalFetchCancelResult::Cancelled { run_id }
        }
        HistoricalFetchCancelOutcome::PublicationInProgress { run_id } => {
            HistoricalFetchCancelResult::PublicationInProgress { run_id }
        }
        HistoricalFetchCancelOutcome::StaleRun {
            requested_run_id,
            active_run_id,
        } => HistoricalFetchCancelResult::StaleRun {
            requested_run_id,
            active_run_id,
        },
        HistoricalFetchCancelOutcome::NoActiveFetch => HistoricalFetchCancelResult::NoActiveFetch,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoricalCaptureStatus {
    pub run_id: u64,
    pub phase: &'static str,
}

pub fn process_historical_capture_status() -> Option<HistoricalCaptureStatus> {
    process_historical_fetch_status().map(|status| HistoricalCaptureStatus {
        run_id: status.run_id,
        phase: status.phase.as_str(),
    })
}

pub fn begin_process_historical_capture()
-> std::result::Result<ProcessHistoricalCapture, HistoricalFetchStartFailure> {
    let queued =
        begin_process_historical_fetch_queued(|| Ok::<_, Infallible>(())).map_err(|error| {
            match error {
                HistoricalFetchQueueStartError::Fetch(
                    HistoricalFetchStartError::AlreadyActive(conflict),
                ) => HistoricalFetchStartFailure::AlreadyActive {
                    active_run_id: conflict.active_run_id,
                },
                HistoricalFetchQueueStartError::Fetch(HistoricalFetchStartError::RunIdOverflow) => {
                    HistoricalFetchStartFailure::RunIdOverflow
                }
                HistoricalFetchQueueStartError::Queue(never) => match never {},
            }
        })?;
    let queued_run_id = queued.run_id();
    if queued.cancellation().is_cancelled() {
        return Err(HistoricalFetchStartFailure::Cancelled {
            run_id: queued_run_id,
        });
    }
    let (active, ()) = queued.into_parts();
    debug_assert_eq!(active.run_id(), queued_run_id);
    Ok(ProcessHistoricalCapture { active })
}

#[cfg(test)]
impl HistoricalPage {
    pub(crate) fn fixture_m1(start_ms: i64, rows: usize) -> Self {
        Self {
            symbol_id: 1,
            timeframe: CanonicalTimeframe::M1,
            bars: (0..rows)
                .map(|row| {
                    let timestamp_ms = start_ms + i64::try_from(row).expect("row") * 60_000;
                    HistoricalBar {
                        timestamp_ms,
                        open: 1.1000,
                        high: 1.1003,
                        low: 1.0997,
                        close: 1.1001,
                        volume: Some(10),
                    }
                })
                .collect(),
            has_more: false,
        }
    }
}
