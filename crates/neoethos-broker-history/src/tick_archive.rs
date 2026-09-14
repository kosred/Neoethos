//! Bounded, resumable raw Bid/Ask acquisition. This is a data archive, NOT a
//! reviewed financial-truth bundle, synchronized quote stream, or trading gate.
//! Reuses the production transport, rate limiter, signed decoder and Vortex IO.

use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use neoethos_core::storage::json::write_json_atomic;
use neoethos_data::core::vortex_io::{read_vortex_array, write_vortex_chunks_fallible_limited};
use neoethos_execution_budget::{
    CancellationToken, CpuPermitRequest, InstalledExecutionBudget, WorkerLimit,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vortex_array::arrays::VarBinArray;
use vortex_array::{IntoArray, ToCanonical};

use crate::ctrader_data::{
    CTraderSymbolInfo, HistoricalTicksResult, parse_symbol_by_id_response,
    parse_symbols_list_response, parse_tick_data_response,
};
use crate::ctrader_historical_admission::HistoricalRequestCancellation;
use crate::ctrader_messages::{
    CTraderOpenApiJsonMessage, CTraderOpenApiSessionResponse, ProductionCTraderOpenApiSession,
    ProductionCTraderOpenApiTransport, build_account_auth_request, build_application_auth_request,
    build_get_tick_data_request, build_symbol_by_id_request, build_symbols_list_request,
    parse_open_api_envelope,
};
use crate::{BrokerEnvironment, load_exact_production_historical_credentials};

const WEEK_MS: i64 = 604_800_000;
const MAX_PAGE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_PAGES: u64 = 250_000;
const SCHEMA: &str = "neoethos.unreviewed-ctrader-tick-archive.v1";
const PAGE_BOUNDARY_POLICY: &str = "defer-oldest-millisecond-and-overlap-next-page-v1";

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Environment {
    Demo,
    Live,
}

impl Environment {
    fn broker(self) -> BrokerEnvironment {
        match self {
            Self::Demo => BrokerEnvironment::Demo,
            Self::Live => BrokerEnvironment::Live,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Download raw, unreviewed cTrader Bid/Ask history; never places orders")]
pub struct TickArchiveCli {
    #[arg(long, value_enum)]
    environment: Environment,
    #[arg(long)]
    account_id: i64,
    #[arg(long)]
    symbol_id: i64,
    #[arg(long)]
    symbol: String,
    #[arg(long, default_value_t = crate::cli::HISTORICAL_START_2016_UNIX_MS)]
    from_ms: i64,
    /// Fixed exclusive end; required so a resume cannot silently change scope.
    #[arg(long)]
    to_ms: i64,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 8 * 1024 * 1024 * 1024)]
    max_archive_bytes: u64,
    #[arg(long, default_value_t = 4 * 1024 * 1024 * 1024)]
    reserve_disk_bytes: u64,
    /// Optional bounded invocation; reaching it is reported as incomplete.
    #[arg(long)]
    max_new_pages: Option<u64>,
    /// Creating this file requests a clean stop at the next page boundary.
    #[arg(long)]
    stop_file: Option<PathBuf>,
    #[arg(long, hide = true)]
    cpu_threads: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    page_boundary_policy: String,
    environment: Environment,
    endpoint: String,
    account_id: i64,
    symbol_id: i64,
    symbol: String,
    from_ms: i64,
    to_ms_exclusive: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Cursor {
    chunk_from_ms: i64,
    chunk_to_ms_exclusive: i64,
    page_to_ms_exclusive: i64,
    side: i32,
}

impl Cursor {
    fn first(plan: &Plan) -> Self {
        let end = (plan.from_ms + WEEK_MS).min(plan.to_ms_exclusive);
        Self {
            chunk_from_ms: plan.from_ms,
            chunk_to_ms_exclusive: end,
            page_to_ms_exclusive: end,
            side: 1,
        }
    }

    fn after(&self, plan: &Plan, page: &HistoricalTicksResult) -> Result<Option<Self>> {
        if page.symbol_id != plan.symbol_id
            || page
                .ticks
                .windows(2)
                .any(|p| p[0].timestamp_ms > p[1].timestamp_ms)
            || page.ticks.iter().any(|t| {
                t.timestamp_ms < self.chunk_from_ms
                    || t.timestamp_ms > self.page_to_ms_exclusive
                    || !t.price.is_finite()
                    || t.price <= 0.0
            })
        {
            bail!("tick page has wrong identity, order, price or requested time bounds");
        }
        if page.has_more {
            let oldest = page
                .ticks
                .first()
                .context("empty tick page with hasMore=true")?
                .timestamp_ms;
            let next_to = oldest.checked_add(1).context("tick cursor overflow")?;
            if next_to >= self.page_to_ms_exclusive {
                bail!(
                    "tick hasMore cannot make older progress: one millisecond may exceed the broker page limit"
                );
            }
            return Ok(Some(Self {
                page_to_ms_exclusive: next_to,
                ..self.clone()
            }));
        }
        if self.side == 1 {
            return Ok(Some(Self {
                page_to_ms_exclusive: self.chunk_to_ms_exclusive,
                side: 2,
                ..self.clone()
            }));
        }
        if self.chunk_to_ms_exclusive == plan.to_ms_exclusive {
            return Ok(None);
        }
        let start = self.chunk_to_ms_exclusive;
        let end = (start + WEEK_MS).min(plan.to_ms_exclusive);
        Ok(Some(Self {
            chunk_from_ms: start,
            chunk_to_ms_exclusive: end,
            page_to_ms_exclusive: end,
            side: 1,
        }))
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PageRecord {
    schema: String,
    plan_sha256: String,
    sequence: u64,
    cursor: Cursor,
    client_msg_id: String,
    captured_at_ms: i64,
    raw_response_sha256: String,
    raw_response_json: String,
}

#[derive(Debug, Serialize)]
pub struct TickArchiveSummary {
    schema: &'static str,
    authority: &'static str,
    pages: u64,
    bid_ticks: u64,
    ask_ticks: u64,
    empty_responses: u64,
    excluded_upper_boundary_ticks: u64,
    deferred_oldest_boundary_ticks: u64,
    oldest_tick_ms: Option<i64>,
    newest_tick_ms: Option<i64>,
    archive_bytes: u64,
    page_hash_chain: String,
    all_requested_windows_visited: bool,
    next: Option<Cursor>,
    stop_reason: String,
}

impl TickArchiveSummary {
    fn new(plan: &Plan) -> Self {
        Self {
            schema: SCHEMA,
            authority: "unreviewed-raw-market-data-only",
            pages: 0,
            bid_ticks: 0,
            ask_ticks: 0,
            empty_responses: 0,
            excluded_upper_boundary_ticks: 0,
            deferred_oldest_boundary_ticks: 0,
            oldest_tick_ms: None,
            newest_tick_ms: None,
            archive_bytes: 0,
            page_hash_chain: String::new(),
            all_requested_windows_visited: false,
            next: Some(Cursor::first(plan)),
            stop_reason: "running".into(),
        }
    }

    fn include(
        &mut self,
        plan: &Plan,
        cursor: &Cursor,
        page: &HistoricalTicksResult,
        bytes: u64,
        hash: &str,
    ) -> Result<()> {
        let next = cursor.after(plan, page)?;
        self.pages += 1;
        self.archive_bytes = self
            .archive_bytes
            .checked_add(bytes)
            .context("archive bytes overflow")?;
        self.empty_responses += u64::from(page.ticks.is_empty());
        // Request the exact upper timestamp, retaining the original response.
        // If the broker includes that boundary, it belongs to the following
        // interval (or preceding pagination page), not twice in the counts.
        for tick in &page.ticks {
            if tick.timestamp_ms == cursor.page_to_ms_exclusive {
                self.excluded_upper_boundary_ticks += 1;
                continue;
            }
            // A capped page may cut THROUGH a group of ticks with the same
            // millisecond. Count none of its oldest group yet. The next
            // request ends at oldest+1ms and retrieves that group in full.
            if page.has_more
                && page
                    .ticks
                    .first()
                    .is_some_and(|t| t.timestamp_ms == tick.timestamp_ms)
            {
                self.deferred_oldest_boundary_ticks += 1;
                continue;
            }
            if cursor.side == 1 {
                self.bid_ticks += 1;
            } else {
                self.ask_ticks += 1;
            }
            self.oldest_tick_ms = Some(
                self.oldest_tick_ms
                    .map_or(tick.timestamp_ms, |old| old.min(tick.timestamp_ms)),
            );
            self.newest_tick_ms = Some(
                self.newest_tick_ms
                    .map_or(tick.timestamp_ms, |old| old.max(tick.timestamp_ms)),
            );
        }
        self.page_hash_chain = sha256(format!("{}:{hash}", self.page_hash_chain).as_bytes());
        self.all_requested_windows_visited = next.is_none();
        self.next = next;
        Ok(())
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut hash = Sha256::new();
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn page_path(root: &Path, seq: u64) -> PathBuf {
    root.join(format!("page-{seq:08}.vortex"))
}

fn has_disk_budget(used: u64, maximum: u64, available: u64, reserve: u64) -> bool {
    maximum.saturating_sub(used) >= MAX_PAGE_BYTES
        && available >= reserve.saturating_add(MAX_PAGE_BYTES * 2)
}

fn decode_record(path: &Path) -> Result<PageRecord> {
    if fs::metadata(path)?.len() > MAX_PAGE_BYTES {
        bail!("oversized archived page");
    }
    let array = read_vortex_array(path)?;
    if array.is_empty() || array.len() > MAX_PAGE_BYTES as usize {
        bail!("archive page has an invalid raw fragment count");
    }
    let view = array.to_varbinview();
    let mut raw = Vec::new();
    for index in 0..view.len() {
        let bytes = view.bytes_at(index);
        if raw.len().saturating_add(bytes.len()) > MAX_PAGE_BYTES as usize {
            bail!("decoded raw page exceeds archive memory bound");
        }
        raw.extend_from_slice(bytes.as_ref());
    }
    serde_json::from_slice(&raw).context("decode archived page")
}

// Multiple bounded UTF-8 fragments let Vortex compress the repeated JSON
// vocabulary; one enormous string prevents useful string-column sampling.
// Concatenation preserves every byte, including multi-byte characters. Old
// one-row pages remain readable and are never rewritten during resume.
fn raw_fragments(mut raw: &str) -> vortex_array::ArrayRef {
    let mut fragments = Vec::new();
    while !raw.is_empty() {
        let mut end = raw.len().min(1024);
        while !raw.is_char_boundary(end) {
            end -= 1;
        }
        fragments.push(&raw[..end]);
        raw = &raw[end..];
    }
    VarBinArray::from(fragments).into_array()
}

fn decode_checked(
    record: &PageRecord,
    plan: &Plan,
    plan_hash: &str,
    cursor: &Cursor,
    sequence: u64,
    symbol: &CTraderSymbolInfo,
) -> Result<HistoricalTicksResult> {
    if record.schema != SCHEMA
        || record.plan_sha256 != plan_hash
        || record.sequence != sequence
        || record.cursor != *cursor
        || record.raw_response_sha256 != sha256(record.raw_response_json.as_bytes())
    {
        bail!("archived page binding, sequence, cursor or raw hash differs");
    }
    let page = parse_tick_data_response(
        &record.raw_response_json,
        plan.account_id,
        &record.client_msg_id,
        symbol,
    )?;
    cursor.after(plan, &page)?;
    Ok(page)
}

fn exchange(
    session: &mut ProductionCTraderOpenApiSession,
    message: &CTraderOpenApiJsonMessage,
    cancel: &HistoricalRequestCancellation,
) -> Result<String> {
    match session.send_one(message, Some(cancel))? {
        CTraderOpenApiSessionResponse::Expected(raw) => Ok(raw),
        CTraderOpenApiSessionResponse::BrokerError(raw) => {
            // Do not print response bodies, which can contain authentication data.
            let code = parse_open_api_envelope(&raw).ok().and_then(|r| {
                r.payload
                    .get("errorCode")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            });
            bail!(
                "cTrader request {} rejected: {}",
                message.payload_type,
                code.as_deref().unwrap_or("unknown broker error")
            )
        }
    }
}

fn require_account(raw: &str, account_id: i64) -> Result<()> {
    if parse_open_api_envelope(raw)?
        .payload
        .get("ctidTraderAccountId")
        .and_then(serde_json::Value::as_i64)
        != Some(account_id)
    {
        bail!("broker response account identity differs");
    }
    Ok(())
}

fn connect(
    plan: &Plan,
    cancel: &HistoricalRequestCancellation,
) -> Result<(ProductionCTraderOpenApiSession, CTraderSymbolInfo, String)> {
    let credentials =
        load_exact_production_historical_credentials(plan.environment.broker(), plan.account_id)
            .map_err(|_| anyhow::anyhow!("exact broker credentials unavailable"))?;
    let mut session =
        ProductionCTraderOpenApiTransport::new(&plan.endpoint).connect_session(Some(cancel))?;
    exchange(
        &mut session,
        &build_application_auth_request(
            &credentials.client_id,
            &credentials.client_secret,
            "tick-archive-app-auth",
        ),
        cancel,
    )
    .map_err(|_| anyhow::anyhow!("tick archive application authentication failed"))?;
    let raw = exchange(
        &mut session,
        &build_account_auth_request(
            plan.account_id,
            &credentials.access_token,
            "tick-archive-account-auth",
        ),
        cancel,
    )
    .map_err(|_| anyhow::anyhow!("tick archive exact account authentication failed"))?;
    require_account(&raw, plan.account_id)?;
    let raw = exchange(
        &mut session,
        &build_symbols_list_request(plan.account_id, false, "tick-archive-symbols"),
        cancel,
    )?;
    require_account(&raw, plan.account_id)?;
    let symbols = parse_symbols_list_response(&raw)?;
    let matches = symbols
        .symbols
        .iter()
        .filter(|s| s.symbol_id == plan.symbol_id)
        .collect::<Vec<_>>();
    if matches.len() != 1 || matches[0].symbol_name != plan.symbol {
        bail!("exact symbol id/name differs for this account");
    }
    let detail = exchange(
        &mut session,
        &build_symbol_by_id_request(
            plan.account_id,
            &[plan.symbol_id],
            "tick-archive-symbol-detail",
        ),
        cancel,
    )?;
    require_account(&detail, plan.account_id)?;
    let mut full = parse_symbol_by_id_response(&detail)?;
    if full.len() != 1 || full[0].symbol_id != plan.symbol_id {
        bail!("full symbol response differs");
    }
    let mut symbol = full.pop().context("full symbol")?;
    symbol.symbol_name = plan.symbol.clone();
    Ok((session, symbol, detail))
}

pub fn execute(
    cli: TickArchiveCli,
    budget: &'static InstalledExecutionBudget,
) -> Result<TickArchiveSummary> {
    if cli.account_id <= 0
        || cli.symbol_id <= 0
        || cli.symbol.trim().is_empty()
        || cli.from_ms < 0
        || cli.from_ms >= cli.to_ms
        || cli.to_ms > chrono::Utc::now().timestamp_millis()
        || cli.to_ms > 2_147_483_646_000
        || cli.max_archive_bytes < MAX_PAGE_BYTES
        || cli.reserve_disk_bytes < MAX_PAGE_BYTES
        || cli.max_new_pages == Some(0)
    {
        bail!("invalid exact tick archive identity, historical interval or resource limits");
    }
    let plan = Plan {
        schema: SCHEMA.into(),
        page_boundary_policy: PAGE_BOUNDARY_POLICY.into(),
        environment: cli.environment,
        endpoint: cli.environment.broker().endpoint_host().into(),
        account_id: cli.account_id,
        symbol_id: cli.symbol_id,
        symbol: cli.symbol.clone(),
        from_ms: cli.from_ms,
        to_ms_exclusive: cli.to_ms,
    };
    let cancel = HistoricalRequestCancellation::new();
    let signal_cancel = cancel.clone();
    let cpu_cancel = CancellationToken::new();
    let signal_cpu = cpu_cancel.clone();
    ctrlc::set_handler(move || {
        signal_cancel.cancel();
        signal_cpu.cancel();
    })?;
    let lease = budget
        .broker()
        .acquire_cancellable(CpuPermitRequest::local(WorkerLimit::new(1)?), &cpu_cancel)?;
    lease.scope(|| run_archive(&cli, &plan, &cancel))
}

fn run_archive(
    cli: &TickArchiveCli,
    plan: &Plan,
    cancel: &HistoricalRequestCancellation,
) -> Result<TickArchiveSummary> {
    fs::create_dir_all(&cli.output)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(cli.output.join("archive.lock"))?;
    fs2::FileExt::try_lock_exclusive(&lock).context("another process owns this archive")?;
    let plan_path = cli.output.join("request.json");
    if plan_path.exists() {
        let saved: Plan = serde_json::from_slice(&fs::read(&plan_path)?)?;
        if saved != *plan {
            bail!("resume requires exactly the same account, symbol and interval");
        }
    } else {
        if fs::read_dir(&cli.output)?
            .filter_map(Result::ok)
            .any(|e| e.file_name() != "archive.lock")
        {
            bail!("new archive directory contains unrelated or unbound files");
        }
        write_json_atomic(&plan_path, plan)?;
    }
    let plan_hash = sha256(&serde_json::to_vec(plan)?);
    let (mut session, symbol, mut summary) =
        prepare_archive_session(cli, plan, &plan_hash, cancel, || connect(plan, cancel))?;
    let initial_pages = summary.pages;
    eprintln!(
        "tick archive resumed: pages={} bid={} ask={} next={:?}",
        summary.pages, summary.bid_ticks, summary.ask_ticks, summary.next
    );
    while let Some(cursor) = summary.next.clone() {
        if cancel.is_cancelled() || cli.stop_file.as_ref().is_some_and(|p| p.exists()) {
            summary.stop_reason = "cancelled-at-page-boundary".into();
            break;
        }
        if cli
            .max_new_pages
            .is_some_and(|n| summary.pages - initial_pages >= n)
        {
            summary.stop_reason = "invocation-page-limit".into();
            break;
        }
        if summary.pages >= MAX_PAGES {
            summary.stop_reason = "archive-page-limit".into();
            break;
        }
        let remaining = cli.max_archive_bytes.saturating_sub(summary.archive_bytes);
        if !has_disk_budget(
            summary.archive_bytes,
            cli.max_archive_bytes,
            fs2::available_space(&cli.output)?,
            cli.reserve_disk_bytes,
        ) {
            summary.stop_reason = "disk-budget-pause".into();
            break;
        }
        let msg_id = format!(
            "tick-archive-{}-{}-{}",
            &plan_hash[..16],
            summary.pages,
            chrono::Utc::now().timestamp_millis()
        );
        let message = build_get_tick_data_request(
            plan.account_id,
            plan.symbol_id,
            cursor.side,
            cursor.chunk_from_ms,
            cursor.page_to_ms_exclusive,
            &msg_id,
        );
        let raw = exchange(&mut session, &message, cancel).with_context(|| {
            format!(
                "tick page {} cursor {cursor:?}; committed pages retained",
                summary.pages
            )
        })?;
        if raw.len() as u64 > MAX_PAGE_BYTES / 2 {
            bail!("raw broker page exceeds bounded archive memory policy");
        }
        let record = PageRecord {
            schema: SCHEMA.into(),
            plan_sha256: plan_hash.clone(),
            sequence: summary.pages,
            cursor: cursor.clone(),
            client_msg_id: msg_id,
            captured_at_ms: chrono::Utc::now().timestamp_millis(),
            raw_response_sha256: sha256(raw.as_bytes()),
            raw_response_json: raw,
        };
        let serialized = serde_json::to_string(&record)?;
        if serialized.len() as u64 > MAX_PAGE_BYTES {
            bail!("serialized page exceeds archive memory bound");
        }
        let page = match decode_checked(&record, plan, &plan_hash, &cursor, summary.pages, &symbol)
        {
            Ok(page) => page,
            Err(error) => {
                let rejected = cli
                    .output
                    .join(format!("rejected-page-{:08}.vortex", summary.pages));
                if !rejected.exists() {
                    write_vortex_chunks_fallible_limited(
                        &rejected,
                        [Ok(raw_fragments(&serialized))],
                        MAX_PAGE_BYTES.min(remaining),
                    )?;
                }
                return Err(error.context(format!(
                    "raw rejected response retained at {}",
                    rejected.display()
                )));
            }
        };
        let path = page_path(&cli.output, summary.pages);
        if path.exists() {
            bail!("refusing to overwrite an existing tick page");
        }
        let stats = write_vortex_chunks_fallible_limited(
            &path,
            [Ok(raw_fragments(&serialized))],
            MAX_PAGE_BYTES.min(remaining),
        )?;
        // Verify the real written bytes before advancing the persistent cursor.
        let reread = decode_record(&path)?;
        decode_checked(&reread, plan, &plan_hash, &cursor, summary.pages, &symbol)?;
        summary.include(plan, &cursor, &page, stats.file_size, &file_hash(&path)?)?;
        write_json_atomic(cli.output.join("progress.json"), &summary)?;
        if summary.pages % 20 == 0 || !page.has_more {
            eprintln!(
                "tick archive pages={} bid={} ask={} bytes={} last_window=[{}, {}) side={} hasMore={}",
                summary.pages,
                summary.bid_ticks,
                summary.ask_ticks,
                summary.archive_bytes,
                cursor.chunk_from_ms,
                cursor.page_to_ms_exclusive,
                cursor.side,
                page.has_more
            );
        }
    }
    if summary.all_requested_windows_visited {
        summary.stop_reason = "requested-window-scan-complete".into();
    }
    write_json_atomic(cli.output.join("progress.json"), &summary)?;
    Ok(summary)
}

fn require_archive_active(
    cancel: &HistoricalRequestCancellation,
    stop_file: Option<&Path>,
) -> Result<()> {
    if cancel.is_cancelled() || stop_file.is_some_and(Path::exists) {
        bail!("tick archive cancelled during resume; committed pages retained");
    }
    Ok(())
}

fn archive_page_count(root: &Path) -> Result<u64> {
    fs::read_dir(root)?.try_fold(0_u64, |count, entry| {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        Ok(count + u64::from(name.starts_with("page-") && name.ends_with(".vortex")))
    })
}

// Full resume validation can take minutes. Keep the network session absent
// during that work: a synchronous request/response socket cannot service
// cTrader's keepalive while it is busy decoding the local archive.
fn prepare_archive_session<S>(
    cli: &TickArchiveCli,
    plan: &Plan,
    plan_hash: &str,
    cancel: &HistoricalRequestCancellation,
    open_session: impl FnOnce() -> Result<(S, CTraderSymbolInfo, String)>,
) -> Result<(S, CTraderSymbolInfo, TickArchiveSummary)> {
    require_archive_active(cancel, cli.stop_file.as_deref())?;
    let symbol_path = cli.output.join("symbol-observation.json");
    let (saved_symbol, summary) = if symbol_path.exists() {
        let raw = fs::read_to_string(&symbol_path)?;
        require_account(&raw, plan.account_id)?;
        let mut symbols = parse_symbol_by_id_response(&raw)?;
        if symbols.len() != 1 || symbols[0].symbol_id != plan.symbol_id {
            bail!("saved symbol observation differs from the exact archive identity");
        }
        let mut saved = symbols.pop().context("saved full symbol")?;
        saved.symbol_name = plan.symbol.clone();
        eprintln!("tick archive validating stored pages before connecting");
        let summary = resume_archive(
            &cli.output,
            plan,
            plan_hash,
            &saved,
            cancel,
            cli.stop_file.as_deref(),
        )?;
        (Some(saved), summary)
    } else {
        if archive_page_count(&cli.output)? != 0 {
            bail!("archived pages have no saved symbol observation; refusing to replace it");
        }
        (None, TickArchiveSummary::new(plan))
    };
    require_archive_active(cancel, cli.stop_file.as_deref())?;
    eprintln!(
        "tick archive verified: pages={} bid={} ask={} hash_chain={}",
        summary.pages, summary.bid_ticks, summary.ask_ticks, summary.page_hash_chain
    );
    eprintln!(
        "tick archive connecting: {} account={} symbol={} requested=[{}, {})",
        plan.endpoint, plan.account_id, plan.symbol, plan.from_ms, plan.to_ms_exclusive
    );
    let (session, symbol, symbol_raw) = open_session()?;
    if symbol.symbol_id != plan.symbol_id || symbol.symbol_name != plan.symbol {
        bail!("connected symbol differs from the exact archive identity");
    }
    if let Some(saved) = saved_symbol {
        // The stored observation is only for local decoding. Fresh authenticated
        // metadata must still agree before any additional page can be acquired.
        if saved.symbol_id != symbol.symbol_id || saved.digits != symbol.digits {
            bail!("symbol decoding identity/digits changed; archive cannot mix them");
        }
    } else {
        // PRESENT-DAY decoding observation, not historical commission/swap.
        // Preserve the first observation; never overwrite it on resume.
        write_json_atomic(
            &symbol_path,
            &serde_json::from_str::<serde_json::Value>(&symbol_raw)?,
        )?;
    }
    Ok((session, symbol, summary))
}

fn resume_archive(
    root: &Path,
    plan: &Plan,
    plan_hash: &str,
    symbol: &CTraderSymbolInfo,
    cancel: &HistoricalRequestCancellation,
    stop_file: Option<&Path>,
) -> Result<TickArchiveSummary> {
    let mut summary = TickArchiveSummary::new(plan);
    // Resume from actual validated pages, not a blindly trusted progress file.
    while page_path(root, summary.pages).exists() {
        require_archive_active(cancel, stop_file)?;
        if summary.pages >= MAX_PAGES {
            bail!("archive page ceiling reached");
        }
        let cursor = summary
            .next
            .clone()
            .context("extra page after interval completion")?;
        let path = page_path(root, summary.pages);
        let record = decode_record(&path)?;
        let page = decode_checked(&record, plan, plan_hash, &cursor, summary.pages, symbol)?;
        summary.include(
            plan,
            &cursor,
            &page,
            fs::metadata(&path)?.len(),
            &file_hash(&path)?,
        )?;
        if summary.pages % 1_000 == 0 {
            eprintln!(
                "tick archive validating: pages={} bid={} ask={}",
                summary.pages, summary.bid_ticks, summary.ask_ticks
            );
        }
    }
    require_archive_active(cancel, stop_file)?;
    let page_count = archive_page_count(root)?;
    if page_count != summary.pages {
        bail!("archive has a page gap; refusing to skip or overwrite data");
    }
    Ok(summary)
}

#[cfg(test)]
mod tests;
