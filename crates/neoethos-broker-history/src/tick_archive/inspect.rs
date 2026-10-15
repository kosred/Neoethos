//! Read-only archive integrity and spread diagnostics. No broker connection,
//! credential access, signal evaluation, financial authority, or trading permit.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use neoethos_execution_budget::{
    CancellationToken, CpuPermitRequest, InstalledExecutionBudget, WorkerLimit,
};
use serde::Serialize;

use super::{
    MAX_PAGES, Plan, TickArchiveSummary, TickOwnership, WEEK_MS, archive_page_count,
    decode_checked, decode_record, file_hash, page_path, require_account, require_archive_active,
    sha256, tick_ownership,
};
use crate::ctrader_data::parse_symbol_by_id_response;
use crate::ctrader_historical_admission::HistoricalRequestCancellation;

const MAX_EVENTS: usize = 10_000_000;
const MAX_POINT_OBSERVATIONS: usize = 1_024;
const MAX_METADATA_BYTES: u64 = 64 * 1024;
const POLICY: &str = "unique-side-price-per-ms-causal-book-strict-age-time-weighted-v1";
const POINT_POLICY: &str = "strictly-before-ms-same-grouped-book-strict-age-v1";

#[derive(Debug, Parser)]
#[command(about = "Verify raw tick archive offline and diagnose spreads; never places orders")]
pub struct TickInspectCli {
    #[arg(long)]
    archive: PathBuf,
    /// Expected hash chain supplied independently; progress.json is never trusted.
    #[arg(long)]
    expected_page_hash_chain: String,
    /// Diagnostic half-open interval, at most seven days. No strategy is evaluated.
    #[arg(long)]
    from_ms: i64,
    #[arg(long)]
    to_ms: i64,
    /// A side is fresh only while age < this limit; both sides must be fresh.
    #[arg(long)]
    max_quote_age_ms: i64,
    /// Hard memory bound for retained ticks, including causal seed padding.
    #[arg(long, default_value_t = 2_000_000)]
    max_events: usize,
    /// Observe the book strictly before this time; repeat in increasing order,
    /// at most 1024 times, within [from-ms, to-ms). Not an execution quote.
    #[arg(long)]
    at_ms: Vec<i64>,
    #[arg(long, hide = true)]
    cpu_threads: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct TickInspectionReport {
    schema: &'static str,
    artifact_class: &'static str,
    promotion_eligibility: &'static str,
    authority: &'static str,
    diagnostic_policy: &'static str,
    point_observation_policy: &'static str,
    binding_sha256: String,
    request_file_sha256: String,
    plan_sha256: String,
    symbol_observation_sha256: String,
    environment: super::Environment,
    account_id: i64,
    symbol_id: i64,
    symbol: String,
    pip_size: f64,
    from_ms: i64,
    to_ms_exclusive: i64,
    max_quote_age_ms: i64,
    retained_window_and_seed_events: usize,
    verified_archive: TickArchiveSummary,
    spreads: SpreadDiagnostics,
    point_observations: Vec<PointObservation>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BookStatus {
    Valid,
    UnavailableOrStale,
    Ambiguous,
    Crossed,
}

#[derive(Debug, Serialize)]
struct PointObservation {
    at_ms: i64,
    status: BookStatus,
    /// Quotes stamped at the observation time are excluded, regardless of side.
    same_timestamp_quotes_excluded: bool,
    bid: Option<PointSide>,
    ask: Option<PointSide>,
}

#[derive(Debug, Serialize)]
struct PointSide {
    timestamp_ms: i64,
    age_ms: i64,
    /// Exact cTrader price units (1 / 100_000). None denotes a tied-price ambiguity.
    price_units: Option<i64>,
}

impl PointObservation {
    fn capture(
        sides: &[Option<SideState>; 2],
        at_ms: i64,
        max_age: i64,
        same_timestamp_quotes_excluded: bool,
    ) -> Self {
        let side = |state: Option<SideState>| {
            state.map(|state| PointSide {
                timestamp_ms: state.timestamp_ms,
                age_ms: at_ms - state.timestamp_ms,
                price_units: state.price_units,
            })
        };
        Self {
            at_ms,
            status: book_status(sides, at_ms, max_age),
            same_timestamp_quotes_excluded,
            bid: side(sides[0]),
            ask: side(sides[1]),
        }
    }
}

#[derive(Debug, Serialize, Default)]
pub struct SpreadDiagnostics {
    interval_ms: u64,
    valid_book_ms: u64,
    unavailable_or_stale_ms: u64,
    ambiguous_book_ms: u64,
    crossed_book_ms: u64,
    valid_book_fraction: f64,
    /// Includes causal seed padding. Repeated identical prices are unambiguous.
    ambiguous_side_timestamp_groups: u64,
    both_sides_updated_timestamp_groups: u64,
    time_weighted_mean_pips: Option<f64>,
    time_weighted_p50_pips: Option<f64>,
    time_weighted_p95_pips: Option<f64>,
    time_weighted_p99_pips: Option<f64>,
    min_pips: Option<f64>,
    max_pips: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct Event {
    timestamp_ms: i64,
    side: i32,
    price_units: i64,
}

#[derive(Clone, Copy)]
struct SideState {
    timestamp_ms: i64,
    // None means multiple distinct prices occurred in this millisecond. Never
    // guess which was last or use an earlier value across this ambiguity.
    price_units: Option<i64>,
}

pub fn execute(
    cli: TickInspectCli,
    budget: &'static InstalledExecutionBudget,
) -> Result<TickInspectionReport> {
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
    lease.scope(|| inspect_archive(&cli, &cancel))
}

fn bounded_metadata(path: &Path) -> Result<Vec<u8>> {
    ensure!(
        fs::metadata(path)?.len() <= MAX_METADATA_BYTES,
        "oversized archive metadata"
    );
    // Read through a bounded handle too, so a concurrent file growth cannot
    // turn the metadata check into an unbounded allocation.
    use std::io::Read;
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_METADATA_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_METADATA_BYTES,
        "oversized archive metadata"
    );
    Ok(bytes)
}

fn inspect_archive(
    cli: &TickInspectCli,
    cancel: &HistoricalRequestCancellation,
) -> Result<TickInspectionReport> {
    ensure!(
        cli.from_ms >= 0
            && cli.to_ms > cli.from_ms
            && cli
                .to_ms
                .checked_sub(cli.from_ms)
                .is_some_and(|span| span <= WEEK_MS)
            && (1..=WEEK_MS).contains(&cli.max_quote_age_ms)
            && (1..=MAX_EVENTS).contains(&cli.max_events)
            && cli.expected_page_hash_chain.len() == 64
            && cli
                .expected_page_hash_chain
                .bytes()
                .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f')),
        "invalid bounded diagnostic interval, freshness, event budget or expected hash chain"
    );
    ensure!(
        cli.at_ms.len() <= MAX_POINT_OBSERVATIONS
            && cli
                .at_ms
                .iter()
                .all(|&at| (cli.from_ms..cli.to_ms).contains(&at))
            && cli.at_ms.windows(2).all(|pair| pair[0] < pair[1]),
        "point observations must be strictly increasing, within the diagnostic interval, and at most 1024"
    );
    // Open the existing lock without creating or changing an archive file.
    let lock = File::open(cli.archive.join("archive.lock"))?;
    fs2::FileExt::try_lock_shared(&lock).context("archive is being modified")?;
    let request_bytes = bounded_metadata(&cli.archive.join("request.json"))?;
    let plan: Plan = serde_json::from_slice(&request_bytes)?;
    ensure!(
        plan.schema == super::SCHEMA
            && plan.page_boundary_policy == super::PAGE_BOUNDARY_POLICY
            && plan.endpoint == plan.environment.broker().endpoint_host()
            && plan.account_id > 0
            && plan.symbol_id > 0
            && !plan.symbol.trim().is_empty()
            && plan.from_ms >= 0
            && plan.from_ms < plan.to_ms_exclusive
            && plan.to_ms_exclusive <= 2_147_483_646_000
            && cli.from_ms >= plan.from_ms
            && cli.to_ms <= plan.to_ms_exclusive,
        "archive plan or diagnostic scope is invalid"
    );
    let plan_sha256 = sha256(&serde_json::to_vec(&plan)?);
    let symbol_bytes = bounded_metadata(&cli.archive.join("symbol-observation.json"))?;
    let symbol_raw = std::str::from_utf8(&symbol_bytes)?;
    require_account(symbol_raw, plan.account_id)?;
    let mut symbols = parse_symbol_by_id_response(symbol_raw)?;
    ensure!(
        symbols.len() == 1 && symbols[0].symbol_id == plan.symbol_id,
        "symbol observation identity differs"
    );
    let mut symbol = symbols.pop().context("missing full symbol observation")?;
    symbol.symbol_name.clone_from(&plan.symbol);
    ensure!(
        (0..=5).contains(&symbol.pip_position),
        "pip precision is unsupported for tick diagnostics"
    );
    let pip_size = 10_f64.powi(-symbol.pip_position);
    let units_per_pip = pip_size * 100_000.0;
    let seed_from = cli
        .from_ms
        .saturating_sub(cli.max_quote_age_ms)
        .max(plan.from_ms);
    let mut events = Vec::with_capacity(cli.max_events);
    let mut summary = TickArchiveSummary::new(&plan);
    while page_path(&cli.archive, summary.pages).exists() {
        require_archive_active(cancel, None)?;
        ensure!(summary.pages < MAX_PAGES, "archive page ceiling reached");
        let cursor = summary
            .next
            .clone()
            .context("extra page after interval completion")?;
        let path = page_path(&cli.archive, summary.pages);
        let record = decode_record(&path)?;
        let page = decode_checked(
            &record,
            &plan,
            &plan_sha256,
            &cursor,
            summary.pages,
            &symbol,
        )?;
        for tick in &page.ticks {
            if tick.timestamp_ms < seed_from
                || tick.timestamp_ms >= cli.to_ms
                || tick_ownership(&cursor, &page, tick) != TickOwnership::Retained
            {
                continue;
            }
            ensure!(
                events.len() < cli.max_events,
                "diagnostic event budget exceeded; narrow the window"
            );
            let scaled = tick.price * 100_000.0;
            ensure!(
                scaled > 0.0
                    && scaled < (1_u64 << 53) as f64
                    && (scaled - scaled.round()).abs() < 0.000_001,
                "tick price cannot be represented in diagnostic units"
            );
            events.push(Event {
                timestamp_ms: tick.timestamp_ms,
                side: cursor.side,
                price_units: scaled.round() as i64,
            });
        }
        summary.include(
            &plan,
            &cursor,
            &page,
            fs::metadata(&path)?.len(),
            &file_hash(&path)?,
        )?;
        if summary.pages % 1_000 == 0 {
            eprintln!(
                "tick inspection verified pages={} bid={} ask={} retained_events={}",
                summary.pages,
                summary.bid_ticks,
                summary.ask_ticks,
                events.len()
            );
        }
    }
    require_archive_active(cancel, None)?;
    ensure!(
        archive_page_count(&cli.archive)? == summary.pages,
        "archive contains a page gap or unexpected page name"
    );
    ensure!(
        summary.all_requested_windows_visited,
        "archive is incomplete; spread diagnostics refused"
    );
    ensure!(
        summary.page_hash_chain == cli.expected_page_hash_chain,
        "archive hash chain differs from the supplied digest"
    );
    summary.stop_reason = "offline-verified-against-expected-hash-chain".into();
    let retained_window_and_seed_events = events.len();
    let (spreads, point_observations) = spread_diagnostics(
        events,
        cli.from_ms,
        cli.to_ms,
        cli.max_quote_age_ms,
        units_per_pip,
        &cli.at_ms,
    );
    require_archive_active(cancel, None)?;
    let request_file_sha256 = sha256(&request_bytes);
    let symbol_observation_sha256 = sha256(&symbol_bytes);
    let binding_sha256 = sha256(&serde_json::to_vec(&(
        POLICY,
        &request_file_sha256,
        &plan_sha256,
        &symbol_observation_sha256,
        &summary.page_hash_chain,
        cli.from_ms,
        cli.to_ms,
        cli.max_quote_age_ms,
        POINT_POLICY,
        &cli.at_ms,
    ))?);
    Ok(TickInspectionReport {
        schema: "neoethos.offline-tick-spread-diagnostics.v2",
        artifact_class: "ResearchOnly",
        promotion_eligibility: "NotPromotionEligible",
        authority: "unreviewed-raw-market-data-diagnostics-only",
        diagnostic_policy: POLICY,
        point_observation_policy: POINT_POLICY,
        binding_sha256,
        request_file_sha256,
        plan_sha256,
        symbol_observation_sha256,
        environment: plan.environment,
        account_id: plan.account_id,
        symbol_id: plan.symbol_id,
        symbol: plan.symbol,
        pip_size,
        from_ms: cli.from_ms,
        to_ms_exclusive: cli.to_ms,
        max_quote_age_ms: cli.max_quote_age_ms,
        retained_window_and_seed_events,
        verified_archive: summary,
        spreads,
        point_observations,
    })
}

fn spread_diagnostics(
    mut events: Vec<Event>,
    from: i64,
    to: i64,
    max_age: i64,
    units_per_pip: f64,
    at_ms: &[i64],
) -> (SpreadDiagnostics, Vec<PointObservation>) {
    // Archive pages run backwards within each side and forwards across weeks.
    // Sorting is only a diagnostic timestamp grouping operation. No ordering
    // between tied side events is asserted and no execution ledger is emitted.
    events.sort_unstable_by_key(|event| (event.timestamp_ms, event.side));
    let mut diagnostics = SpreadDiagnostics {
        interval_ms: (to - from) as u64,
        ..Default::default()
    };
    let mut histogram = BTreeMap::<i64, u64>::new();
    let mut sides: [Option<SideState>; 2] = [None, None];
    let mut last = from;
    let mut index = 0;
    let mut pending = at_ms.iter().copied().peekable();
    let mut observations = Vec::with_capacity(at_ms.len());
    while index < events.len() {
        let timestamp = events[index].timestamp_ms;
        // Observe before applying the entire timestamp group. Side ordering in
        // archived pages cannot prove ordering relative to a broker execution.
        while pending.peek().is_some_and(|&at| at <= timestamp) {
            let at = pending.next().unwrap();
            observations.push(PointObservation::capture(
                &sides,
                at,
                max_age,
                at == timestamp,
            ));
        }
        let at = timestamp.max(from).min(to);
        include_interval(&mut diagnostics, &mut histogram, &sides, last, at, max_age);
        last = at;
        let mut changed = [false; 2];
        while index < events.len() && events[index].timestamp_ms == timestamp {
            let side = (events[index].side - 1) as usize;
            let first = events[index].price_units;
            let mut unique = true;
            while index < events.len()
                && events[index].timestamp_ms == timestamp
                && (events[index].side - 1) as usize == side
            {
                unique &= events[index].price_units == first;
                index += 1;
            }
            diagnostics.ambiguous_side_timestamp_groups += u64::from(!unique);
            sides[side] = Some(SideState {
                timestamp_ms: timestamp,
                price_units: unique.then_some(first),
            });
            changed[side] = true;
        }
        diagnostics.both_sides_updated_timestamp_groups += u64::from(changed[0] && changed[1]);
    }
    include_interval(&mut diagnostics, &mut histogram, &sides, last, to, max_age);
    observations.extend(pending.map(|at| PointObservation::capture(&sides, at, max_age, false)));
    diagnostics.valid_book_fraction =
        diagnostics.valid_book_ms as f64 / diagnostics.interval_ms as f64;
    if diagnostics.valid_book_ms > 0 {
        diagnostics.time_weighted_mean_pips = Some(
            histogram
                .iter()
                .map(|(&units, &ms)| units as f64 * ms as f64)
                .sum::<f64>()
                / diagnostics.valid_book_ms as f64
                / units_per_pip,
        );
        let quantile = |percent: u64| {
            let rank = (diagnostics.valid_book_ms * percent).div_ceil(100);
            let mut accumulated = 0;
            histogram.iter().find_map(|(&units, &ms)| {
                accumulated += ms;
                (accumulated >= rank).then_some(units as f64 / units_per_pip)
            })
        };
        diagnostics.time_weighted_p50_pips = quantile(50);
        diagnostics.time_weighted_p95_pips = quantile(95);
        diagnostics.time_weighted_p99_pips = quantile(99);
        diagnostics.min_pips = histogram
            .first_key_value()
            .map(|(&units, _)| units as f64 / units_per_pip);
        diagnostics.max_pips = histogram
            .last_key_value()
            .map(|(&units, _)| units as f64 / units_per_pip);
    }
    (diagnostics, observations)
}

fn book_status(sides: &[Option<SideState>; 2], at: i64, max_age: i64) -> BookStatus {
    let [Some(bid), Some(ask)] = sides else {
        return BookStatus::UnavailableOrStale;
    };
    if at >= bid.timestamp_ms.saturating_add(max_age)
        || at >= ask.timestamp_ms.saturating_add(max_age)
    {
        return BookStatus::UnavailableOrStale;
    }
    match (bid.price_units, ask.price_units) {
        (Some(bid), Some(ask)) if ask < bid => BookStatus::Crossed,
        (Some(_), Some(_)) => BookStatus::Valid,
        _ => BookStatus::Ambiguous,
    }
}

fn include_interval(
    diagnostics: &mut SpreadDiagnostics,
    histogram: &mut BTreeMap<i64, u64>,
    sides: &[Option<SideState>; 2],
    from: i64,
    to: i64,
    max_age: i64,
) {
    if from >= to {
        return;
    }
    let duration = (to - from) as u64;
    let [Some(bid), Some(ask)] = sides else {
        diagnostics.unavailable_or_stale_ms += duration;
        return;
    };
    let fresh_until = bid
        .timestamp_ms
        .saturating_add(max_age)
        .min(ask.timestamp_ms.saturating_add(max_age))
        .min(to)
        .max(from);
    let fresh_ms = (fresh_until - from) as u64;
    diagnostics.unavailable_or_stale_ms += duration - fresh_ms;
    if fresh_ms == 0 {
        return;
    }
    match book_status(sides, from, max_age) {
        BookStatus::Ambiguous => diagnostics.ambiguous_book_ms += fresh_ms,
        BookStatus::Crossed => diagnostics.crossed_book_ms += fresh_ms,
        BookStatus::Valid => {
            diagnostics.valid_book_ms += fresh_ms;
            let spread = ask.price_units.unwrap() - bid.price_units.unwrap();
            *histogram.entry(spread).or_default() += fresh_ms;
        }
        BookStatus::UnavailableOrStale => unreachable!("fresh interval has both sides"),
    }
}

#[cfg(test)]
mod tests;
