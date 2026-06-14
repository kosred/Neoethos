use crate::app_services::bootstrap_writer::write_bootstrap_vortex;
use crate::app_services::ctrader_data::{CTraderChartHistoryRequest, load_historical_bars_only};
use anyhow::{Context, Result, bail};
use neoethos_data::load_symbol_timeframe;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapChunk {
    pub from_timestamp_ms: i64,
    pub to_timestamp_ms: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedBar {
    pub timestamp_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageSegment {
    pub from_timestamp_ns: i64,
    pub to_timestamp_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalCoverageReport {
    pub requested_from_timestamp_ns: i64,
    pub requested_to_timestamp_ns: i64,
    pub covered_segments: Vec<CoverageSegment>,
    pub missing_segments: Vec<CoverageSegment>,
    pub fully_covered: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapOutcome {
    pub output_path: PathBuf,
    pub coverage: LocalCoverageReport,
    pub sources_used: Vec<String>,
    pub warnings: Vec<String>,
    pub bars_written: usize,
}

// `dead_code` until the live bootstrap path requests historical chunks through
// this planner (Phase 2-5). The chunking logic is fully tested below.
#[allow(dead_code)]
pub fn plan_bootstrap_chunks(
    now_timestamp_ms: i64,
    timeframe: &str,
    years: u32,
) -> Result<Vec<BootstrapChunk>> {
    if years == 0 {
        bail!("years must be positive");
    }

    let chunk_span_ms = match timeframe.trim().to_ascii_uppercase().as_str() {
        "M1" => 14 * DAY_MS,
        "M5" => 30 * DAY_MS,
        "M15" => 90 * DAY_MS,
        "H1" => 180 * DAY_MS,
        "H4" => 365 * DAY_MS,
        "D1" => 365 * DAY_MS,
        other => bail!("unsupported timeframe: {}", other),
    };

    let requested_span_ms = years as i64 * 365 * DAY_MS;
    let start_timestamp_ms = now_timestamp_ms.saturating_sub(requested_span_ms);
    let mut chunks = Vec::new();
    let mut cursor = start_timestamp_ms;
    while cursor < now_timestamp_ms {
        let to_timestamp_ms = (cursor + chunk_span_ms).min(now_timestamp_ms);
        chunks.push(BootstrapChunk {
            from_timestamp_ms: cursor,
            to_timestamp_ms,
        });
        cursor = to_timestamp_ms;
    }
    Ok(chunks)
}

pub fn clean_normalized_bars(bars: &[NormalizedBar]) -> Result<Vec<NormalizedBar>> {
    let mut cleaned = bars.to_vec();
    cleaned.sort_by_key(|bar| bar.timestamp_ns);
    cleaned.dedup_by_key(|bar| bar.timestamp_ns);

    for bar in &cleaned {
        if !bar.open.is_finite()
            || !bar.high.is_finite()
            || !bar.low.is_finite()
            || !bar.close.is_finite()
            || !bar.volume.is_finite()
        {
            bail!("non-finite OHLCV value detected");
        }

        if bar.volume < 0.0 {
            bail!("negative volume detected");
        }

        if bar.high < bar.low
            || bar.open < bar.low
            || bar.open > bar.high
            || bar.close < bar.low
            || bar.close > bar.high
        {
            bail!("invalid OHLC row detected");
        }
    }

    Ok(cleaned)
}

pub fn trailing_year_range_ns(now_timestamp_ms: i64, years: u32) -> Result<(i64, i64)> {
    if years == 0 {
        bail!("years must be positive");
    }

    let end_ns = now_timestamp_ms
        .checked_mul(1_000_000)
        .context("failed to convert bootstrap end timestamp to ns")?;
    let start_ns = end_ns.saturating_sub(years as i64 * 365 * DAY_NS);
    Ok((start_ns, end_ns))
}

pub fn inspect_local_bar_coverage(
    data_root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
    requested_from_timestamp_ns: i64,
    requested_to_timestamp_ns: i64,
) -> Result<LocalCoverageReport> {
    let step_ns = timeframe_step_ns(timeframe)?;
    let ohlcv = load_symbol_timeframe(data_root, symbol, timeframe)
        .with_context(|| format!("failed to load local coverage dataset {symbol} {timeframe}"))?;
    // `vortex_array_to_ohlcv` normalises stored timestamps to milliseconds at
    // the load boundary (via `normalize_timestamps_to_inferred_millis`). The
    // bootstrap coverage pipeline works throughout in nanoseconds: step_ns,
    // is_fx_trading_timestamp, and CoverageSegment fields are all nanosecond
    // quantities. Multiply back to nanoseconds so the rest of the function is
    // unit-consistent with the NormalizedBar timestamps written by the writer.
    let raw_ms = ohlcv.timestamp.context("local dataset has no timestamps")?;
    // #157: checked_mul instead of saturating_mul. The original code
    // silently clamped any ms > 9_223_372_036_854 (year 2262) to
    // i64::MAX, which collapsed the entire coverage window into one
    // saturated bucket — silent corruption. Real timestamps from the
    // broker are nowhere near 2262, so a future-dated value here is
    // almost certainly upstream corruption that deserves a loud
    // error rather than a mis-bucketed integral overflow.
    let mut timestamps: Vec<i64> = raw_ms
        .into_iter()
        .map(|ms| {
            ms.checked_mul(1_000_000).ok_or_else(|| {
                anyhow::anyhow!(
                    "timestamp {} ms × 1_000_000 overflows i64 nanoseconds — \
                     refusing to silently saturate; check upstream data source",
                    ms
                )
            })
        })
        .collect::<Result<Vec<i64>>>()?;
    timestamps.sort_unstable();
    timestamps.dedup();

    let relevant: Vec<i64> = timestamps
        .into_iter()
        .filter(|ts| *ts >= requested_from_timestamp_ns && *ts < requested_to_timestamp_ns)
        .collect();

    let covered_segments = contiguous_segments(&relevant, step_ns);
    let missing_segments = missing_segments(
        requested_from_timestamp_ns,
        requested_to_timestamp_ns,
        &covered_segments,
        step_ns,
    );
    let fully_covered = !covered_segments.is_empty() && missing_segments.is_empty();

    Ok(LocalCoverageReport {
        requested_from_timestamp_ns,
        requested_to_timestamp_ns,
        covered_segments,
        missing_segments,
        fully_covered,
    })
}

pub fn inspect_local_bar_coverage_or_empty(
    data_root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
    requested_from_timestamp_ns: i64,
    requested_to_timestamp_ns: i64,
) -> Result<LocalCoverageReport> {
    match inspect_local_bar_coverage(
        data_root,
        symbol,
        timeframe,
        requested_from_timestamp_ns,
        requested_to_timestamp_ns,
    ) {
        Ok(report) => Ok(report),
        Err(err) if looks_like_missing_local_dataset(&err) => Ok(LocalCoverageReport {
            requested_from_timestamp_ns,
            requested_to_timestamp_ns,
            covered_segments: Vec::new(),
            missing_segments: vec![CoverageSegment {
                from_timestamp_ns: requested_from_timestamp_ns,
                to_timestamp_ns: requested_to_timestamp_ns,
            }],
            fully_covered: false,
        }),
        Err(err) => Err(err),
    }
}

#[allow(dead_code)]
pub fn bootstrap_with_fetcher<F>(
    data_root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
    now_timestamp_ms: i64,
    years: u32,
    fetcher: F,
) -> Result<BootstrapOutcome>
where
    F: FnMut(i64, i64) -> Result<Vec<NormalizedBar>>,
{
    let (requested_from_ns, requested_to_ns) = trailing_year_range_ns(now_timestamp_ms, years)?;
    bootstrap_with_fetcher_for_range(
        data_root,
        symbol,
        timeframe,
        requested_from_ns,
        requested_to_ns,
        fetcher,
    )
}

fn bootstrap_with_fetcher_for_range<F>(
    data_root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
    requested_from_ns: i64,
    requested_to_ns: i64,
    mut fetcher: F,
) -> Result<BootstrapOutcome>
where
    F: FnMut(i64, i64) -> Result<Vec<NormalizedBar>>,
{
    let data_root = data_root.as_ref();
    let mut coverage = inspect_local_bar_coverage_or_empty(
        data_root,
        symbol,
        timeframe,
        requested_from_ns,
        requested_to_ns,
    )?;
    let mut fetched_bars = Vec::new();
    let mut sources_used = Vec::new();

    if !coverage.fully_covered {
        for missing in coverage.missing_segments.clone() {
            let chunk_requests = plan_missing_segment_chunks(
                timeframe,
                missing.from_timestamp_ns,
                missing.to_timestamp_ns,
            )?;
            for chunk in chunk_requests {
                let mut bars = fetcher(chunk.from_timestamp_ms, chunk.to_timestamp_ms)
                    .with_context(|| {
                        format!(
                            "failed to fetch bootstrap chunk {} {} [{}..{})",
                            symbol, timeframe, chunk.from_timestamp_ms, chunk.to_timestamp_ms
                        )
                    })?;
                fetched_bars.append(&mut bars);
            }
        }
        if !fetched_bars.is_empty() {
            let merged =
                merge_existing_and_fetched_bars(data_root, symbol, timeframe, &fetched_bars)?;
            let output_path = write_bootstrap_vortex(data_root, symbol, timeframe, &merged)?;
            coverage = inspect_local_bar_coverage_or_empty(
                data_root,
                symbol,
                timeframe,
                requested_from_ns,
                requested_to_ns,
            )?;
            sources_used.push("RemoteBootstrap".to_string());
            return Ok(BootstrapOutcome {
                output_path,
                coverage,
                sources_used,
                warnings: Vec::new(),
                bars_written: merged.len(),
            });
        }
    }

    let output_path =
        crate::app_services::bootstrap_writer::bootstrap_vortex_path(data_root, symbol, timeframe);
    Ok(BootstrapOutcome {
        output_path,
        coverage,
        sources_used,
        warnings: Vec::new(),
        bars_written: 0,
    })
}

#[allow(dead_code)]
pub fn bootstrap_from_ctrader_history(
    data_root: impl AsRef<Path>,
    request: &CTraderChartHistoryRequest,
    now_timestamp_ms: i64,
    years: u32,
) -> Result<BootstrapOutcome> {
    bootstrap_with_fetcher(
        data_root,
        &request.symbol_name,
        &request.timeframe,
        now_timestamp_ms,
        years,
        |from_timestamp_ms, to_timestamp_ms| {
            let result = load_historical_bars_only(&CTraderChartHistoryRequest {
                client_id: request.client_id.clone(),
                client_secret: request.client_secret.clone(),
                access_token: request.access_token.clone(),
                environment: request.environment,
                account_id: request.account_id.clone(),
                symbol_name: request.symbol_name.clone(),
                timeframe: request.timeframe.clone(),
                from_timestamp_ms,
                to_timestamp_ms,
                count: request.count,
            })?;
            Ok(result
                .bars
                .into_iter()
                .map(|bar| NormalizedBar {
                    timestamp_ns: bar.timestamp_ms * 1_000_000,
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                    volume: bar.volume.unwrap_or_default() as f64,
                })
                .collect())
        },
    )
}

fn contiguous_segments(timestamps: &[i64], step_ns: i64) -> Vec<CoverageSegment> {
    if timestamps.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut start = timestamps[0];
    let mut prev = timestamps[0];

    for &ts in &timestamps[1..] {
        if ts - prev > step_ns {
            segments.push(CoverageSegment {
                from_timestamp_ns: start,
                to_timestamp_ns: prev + step_ns,
            });
            start = ts;
        }
        prev = ts;
    }

    segments.push(CoverageSegment {
        from_timestamp_ns: start,
        to_timestamp_ns: prev + step_ns,
    });
    segments
}

fn missing_segments(
    requested_from_timestamp_ns: i64,
    requested_to_timestamp_ns: i64,
    covered_segments: &[CoverageSegment],
    step_ns: i64,
) -> Vec<CoverageSegment> {
    let mut missing = Vec::new();
    let mut cursor = requested_from_timestamp_ns;

    for segment in covered_segments {
        if segment.from_timestamp_ns > cursor {
            push_missing_segment_if_trading(
                &mut missing,
                cursor,
                segment.from_timestamp_ns,
                step_ns,
            );
        }
        cursor = cursor.max(segment.to_timestamp_ns);
    }

    if cursor < requested_to_timestamp_ns {
        push_missing_segment_if_trading(&mut missing, cursor, requested_to_timestamp_ns, step_ns);
    }

    missing
}

fn push_missing_segment_if_trading(
    missing: &mut Vec<CoverageSegment>,
    from_timestamp_ns: i64,
    to_timestamp_ns: i64,
    step_ns: i64,
) {
    if from_timestamp_ns >= to_timestamp_ns
        || is_fx_weekend_gap_only(from_timestamp_ns, to_timestamp_ns, step_ns)
    {
        return;
    }

    missing.push(CoverageSegment {
        from_timestamp_ns,
        to_timestamp_ns,
    });
}

fn is_fx_weekend_gap_only(from_timestamp_ns: i64, to_timestamp_ns: i64, step_ns: i64) -> bool {
    if from_timestamp_ns >= to_timestamp_ns || step_ns <= 0 {
        return false;
    }

    let mut cursor = from_timestamp_ns;
    while cursor < to_timestamp_ns {
        if is_fx_trading_timestamp(cursor) {
            return false;
        }
        cursor = cursor.saturating_add(step_ns);
    }

    true
}

fn is_fx_trading_timestamp(timestamp_ns: i64) -> bool {
    let minutes_of_day = timestamp_ns
        .rem_euclid(DAY_NS)
        .div_euclid(60 * 1_000_000_000);
    let weekday = (timestamp_ns.div_euclid(DAY_NS) + 3).rem_euclid(7);

    match weekday {
        0..=4 => {
            (TRADING_SESSION_START_MINUTES..=TRADING_SESSION_END_MINUTES).contains(&minutes_of_day)
        }
        _ => false,
    }
}

fn timeframe_step_ns(timeframe: &str) -> Result<i64> {
    let minutes = neoethos_data::parse_timeframe_to_minutes(timeframe)?;
    Ok(minutes * 60 * 1_000_000_000)
}

fn plan_missing_segment_chunks(
    timeframe: &str,
    from_timestamp_ns: i64,
    to_timestamp_ns: i64,
) -> Result<Vec<BootstrapChunk>> {
    let chunk_span_ms = bootstrap_chunk_span_ms(timeframe)?;
    let mut chunks = Vec::new();
    let mut cursor_ms = from_timestamp_ns / 1_000_000;
    let to_timestamp_ms = to_timestamp_ns / 1_000_000;
    while cursor_ms < to_timestamp_ms {
        let next_to_ms = (cursor_ms + chunk_span_ms).min(to_timestamp_ms);
        chunks.push(BootstrapChunk {
            from_timestamp_ms: cursor_ms,
            to_timestamp_ms: next_to_ms,
        });
        cursor_ms = next_to_ms;
    }
    Ok(chunks)
}

fn bootstrap_chunk_span_ms(timeframe: &str) -> Result<i64> {
    match timeframe.trim().to_ascii_uppercase().as_str() {
        "M1" => Ok(14 * DAY_MS),
        "M5" => Ok(30 * DAY_MS),
        "M15" => Ok(90 * DAY_MS),
        "H1" => Ok(180 * DAY_MS),
        "H4" | "D1" => Ok(365 * DAY_MS),
        other => bail!("unsupported timeframe: {}", other),
    }
}

fn merge_existing_and_fetched_bars(
    data_root: &Path,
    symbol: &str,
    timeframe: &str,
    fetched_bars: &[NormalizedBar],
) -> Result<Vec<NormalizedBar>> {
    let mut combined = load_existing_normalized_bars(data_root, symbol, timeframe)?;
    combined.extend_from_slice(fetched_bars);
    clean_normalized_bars(&combined)
}

fn load_existing_normalized_bars(
    data_root: &Path,
    symbol: &str,
    timeframe: &str,
) -> Result<Vec<NormalizedBar>> {
    match load_symbol_timeframe(data_root, symbol, timeframe) {
        Ok(ohlcv) => Ok(ohlcv
            .timestamp
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(idx, timestamp_ms)| {
                // `load_symbol_timeframe` normalises timestamps to milliseconds at
                // the vortex-load boundary (via `normalize_timestamps_to_inferred_millis`).
                // `NormalizedBar.timestamp_ns` is a nanosecond field; multiply back so
                // the merge step, `clean_normalized_bars`, and the subsequent
                // `inspect_local_bar_coverage` call all operate on a consistent unit.
                let timestamp_ns = timestamp_ms.saturating_mul(1_000_000);
                NormalizedBar {
                    timestamp_ns,
                    open: ohlcv.open[idx],
                    high: ohlcv.high[idx],
                    low: ohlcv.low[idx],
                    close: ohlcv.close[idx],
                    volume: ohlcv
                        .volume
                        .as_ref()
                        .and_then(|values| values.get(idx).copied())
                        .unwrap_or_default(),
                }
            })
            .collect()),
        Err(err) if looks_like_missing_local_dataset(&err) => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

fn looks_like_missing_local_dataset(err: &anyhow::Error) -> bool {
    let text = err.to_string();
    text.contains("path not found")
        || text.contains("no vortex files found")
        || text.contains("vortex dataset not found")
}

const DAY_MS: i64 = 24 * 60 * 60 * 1000;
const DAY_NS: i64 = DAY_MS * 1_000_000;
const TRADING_SESSION_START_MINUTES: i64 = 5;
const TRADING_SESSION_END_MINUTES: i64 = 23 * 60 + 55;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::bootstrap_writer::bootstrap_vortex_path;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const ONE_MINUTE_NS: i64 = 60 * 1_000_000_000;

    fn unique_temp_root(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "neoethos_app_ctrader_bootstrap_{}_{}_{}",
            test_name,
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn clean_normalized_bars_sorts_and_deduplicates_by_timestamp() {
        let cleaned = clean_normalized_bars(&[
            NormalizedBar {
                timestamp_ns: 3,
                open: 1.3,
                high: 1.4,
                low: 1.2,
                close: 1.35,
                volume: 2.0,
            },
            NormalizedBar {
                timestamp_ns: 1,
                open: 1.1,
                high: 1.2,
                low: 1.0,
                close: 1.15,
                volume: 1.0,
            },
            NormalizedBar {
                timestamp_ns: 1,
                open: 9.9,
                high: 9.9,
                low: 9.9,
                close: 9.9,
                volume: 9.9,
            },
            NormalizedBar {
                timestamp_ns: 2,
                open: 1.2,
                high: 1.3,
                low: 1.1,
                close: 1.25,
                volume: 1.5,
            },
        ])
        .expect("cleaned bars");

        assert_eq!(
            cleaned
                .iter()
                .map(|bar| bar.timestamp_ns)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!((cleaned[0].open - 1.1).abs() < f64::EPSILON);
    }

    #[test]
    fn clean_normalized_bars_rejects_invalid_ohlc_rows() {
        let err = clean_normalized_bars(&[NormalizedBar {
            timestamp_ns: 1,
            open: 1.5,
            high: 1.2,
            low: 1.0,
            close: 1.1,
            volume: 1.0,
        }])
        .expect_err("invalid ohlc row must fail");

        assert!(err.to_string().contains("OHLC"));
    }

    #[test]
    fn local_coverage_detects_full_requested_window() {
        let root = unique_temp_root("full_coverage");
        let start_ns = 1_700_000_000_000_000_000;
        let bars = (0..5)
            .map(|idx| NormalizedBar {
                timestamp_ns: start_ns + idx * ONE_MINUTE_NS,
                open: 1.1,
                high: 1.2,
                low: 1.0,
                close: 1.15,
                volume: 1.0,
            })
            .collect::<Vec<_>>();
        write_bootstrap_vortex(&root, "EURUSD", "M1", &bars).expect("write local vortex");

        let report = inspect_local_bar_coverage(
            &root,
            "EURUSD",
            "M1",
            start_ns,
            start_ns + 5 * ONE_MINUTE_NS,
        )
        .expect("coverage report");

        assert!(report.fully_covered);
        assert_eq!(report.covered_segments.len(), 1);
        assert!(report.missing_segments.is_empty());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn local_coverage_reports_gap_between_contiguous_runs() {
        let root = unique_temp_root("gap_coverage");
        let start_ns = 1_700_000_000_000_000_000;
        let bars = vec![0_i64, 1, 2, 5, 6]
            .into_iter()
            .map(|idx| NormalizedBar {
                timestamp_ns: start_ns + idx * ONE_MINUTE_NS,
                open: 1.1,
                high: 1.2,
                low: 1.0,
                close: 1.15,
                volume: 1.0,
            })
            .collect::<Vec<_>>();
        write_bootstrap_vortex(&root, "EURUSD", "M1", &bars).expect("write local vortex");

        let report = inspect_local_bar_coverage(
            &root,
            "EURUSD",
            "M1",
            start_ns,
            start_ns + 7 * ONE_MINUTE_NS,
        )
        .expect("coverage report");

        assert!(!report.fully_covered);
        assert_eq!(report.covered_segments.len(), 2);
        assert_eq!(report.missing_segments.len(), 1);
        assert_eq!(
            report.missing_segments[0],
            CoverageSegment {
                from_timestamp_ns: start_ns + 3 * ONE_MINUTE_NS,
                to_timestamp_ns: start_ns + 5 * ONE_MINUTE_NS,
            }
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn local_coverage_ignores_weekend_only_gap() {
        let root = unique_temp_root("weekend_gap");
        let friday_close_ns = 1_774_655_700_000_i64 * 1_000_000;
        let monday_open_ns = 1_774_829_100_000_i64 * 1_000_000;
        let bars = vec![
            NormalizedBar {
                timestamp_ns: friday_close_ns,
                open: 1.1,
                high: 1.2,
                low: 1.0,
                close: 1.15,
                volume: 1.0,
            },
            NormalizedBar {
                timestamp_ns: monday_open_ns,
                open: 1.2,
                high: 1.3,
                low: 1.1,
                close: 1.25,
                volume: 1.0,
            },
        ];
        write_bootstrap_vortex(&root, "EURUSD", "M1", &bars).expect("write local vortex");

        let report = inspect_local_bar_coverage(
            &root,
            "EURUSD",
            "M1",
            friday_close_ns,
            monday_open_ns + ONE_MINUTE_NS,
        )
        .expect("coverage report");

        assert!(report.fully_covered);
        assert!(report.missing_segments.is_empty());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn target_path_uses_expected_layout() {
        let path = bootstrap_vortex_path(Path::new("data"), "eurusd", "m15");

        assert_eq!(
            path,
            PathBuf::from("data")
                .join("symbol=EURUSD")
                .join("timeframe=M15")
                .join("data.vortex")
        );
    }

    #[test]
    fn trailing_year_range_uses_trailing_utc_days() {
        let now_ms = 1_700_000_000_000;

        let (start_ns, end_ns) = trailing_year_range_ns(now_ms, 2).expect("trailing range");

        assert_eq!(end_ns, now_ms * 1_000_000);
        assert_eq!(end_ns - start_ns, 2 * 365 * DAY_NS);
    }

    #[test]
    fn bootstrap_with_fetcher_skips_remote_when_local_data_already_covers_request() {
        let root = unique_temp_root("skip_remote");
        let start_ns = 1_700_000_000_000_000_000;
        let bars = (0..5)
            .map(|idx| NormalizedBar {
                timestamp_ns: start_ns + idx * ONE_MINUTE_NS,
                open: 1.1,
                high: 1.2,
                low: 1.0,
                close: 1.15,
                volume: 1.0,
            })
            .collect::<Vec<_>>();
        write_bootstrap_vortex(&root, "EURUSD", "M1", &bars).expect("write local vortex");

        let mut called = false;
        let outcome = bootstrap_with_fetcher_for_range(
            &root,
            "EURUSD",
            "M1",
            start_ns,
            start_ns + 5 * ONE_MINUTE_NS,
            |_, _| {
                called = true;
                Ok(Vec::new())
            },
        )
        .expect("bootstrap outcome");

        assert!(!called);
        assert!(outcome.coverage.fully_covered);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn bootstrap_with_fetcher_merges_remote_bars_into_missing_local_gap() {
        let root = unique_temp_root("merge_remote");
        let start_ns = 1_700_000_000_000_000_000;
        let local_bars = vec![0_i64, 1, 4]
            .into_iter()
            .map(|idx| NormalizedBar {
                timestamp_ns: start_ns + idx * ONE_MINUTE_NS,
                open: 1.1,
                high: 1.2,
                low: 1.0,
                close: 1.15,
                volume: 1.0,
            })
            .collect::<Vec<_>>();
        write_bootstrap_vortex(&root, "EURUSD", "M1", &local_bars).expect("write local vortex");

        let outcome = bootstrap_with_fetcher_for_range(
            &root,
            "EURUSD",
            "M1",
            start_ns,
            start_ns + 5 * ONE_MINUTE_NS,
            |from_ms, to_ms| {
                let from_ns = from_ms * 1_000_000;
                let to_ns = to_ms * 1_000_000;
                let mut bars = Vec::new();
                let mut ts = from_ns;
                while ts < to_ns {
                    if ts == start_ns + 2 * ONE_MINUTE_NS || ts == start_ns + 3 * ONE_MINUTE_NS {
                        bars.push(NormalizedBar {
                            timestamp_ns: ts,
                            open: 1.2,
                            high: 1.3,
                            low: 1.1,
                            close: 1.25,
                            volume: 2.0,
                        });
                    }
                    ts += ONE_MINUTE_NS;
                }
                Ok(bars)
            },
        )
        .expect("bootstrap outcome");

        assert!(outcome.coverage.fully_covered);
        assert_eq!(outcome.bars_written, 5);
        assert_eq!(outcome.sources_used, vec!["RemoteBootstrap".to_string()]);

        let _ = std::fs::remove_dir_all(root);
    }
}
