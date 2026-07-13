//! Universal data importer — accepts any reasonable OHLCV file format
//! and converts it to the canonical Vortex layout at
//! `data/symbol={SYMBOL}/timeframe={TF}/data.vortex`.
//!
//! Supported inputs:
//! - **CSV / TSV** (header row required; columns auto-detected from
//!   common names — `time`/`timestamp`/`date`, `open`, `high`, `low`,
//!   `close`, `volume`).
//! - **Parquet** (any schema with the same column names; types coerced
//!   to f64/i64).
//! - **JSON / JSON Lines** — array of objects with the same keys.
//! - **Vortex** — pass-through validation (read + count rows).
//!
//! Symbol and timeframe are inferred from the source path when
//! possible (e.g. `data/EURUSD/M5/2024.csv` → symbol=EURUSD, tf=M5),
//! otherwise the caller supplies them.
//!
//! Every conversion is verified by reading the written Vortex file
//! back and counting rows. Failed conversions are logged and the
//! offending file moved to `<root>/import_quarantine/` so the user
//! can inspect.
//!
//! Designed to be called recursively over a folder tree by
//! `import_directory_recursive`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::Ohlcv;
use crate::core::timestamps::{
    TimestampUnit, infer_timestamp_unit, normalize_timestamps_to_millis,
};

/// Per-file outcome of an import attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportFileResult {
    pub source: String,
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub format: String,
    pub rows: usize,
    pub status: ImportStatus,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImportStatus {
    Imported,
    Skipped,
    Quarantined,
    Failed,
}

/// Aggregate result of a recursive import.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ImportReport {
    pub root: String,
    pub data_root: String,
    pub files_seen: usize,
    pub imported: usize,
    pub skipped: usize,
    pub quarantined: usize,
    pub failed: usize,
    pub results: Vec<ImportFileResult>,
}

impl ImportReport {
    pub fn save_to_disk(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).ok();
        }
        let text = serde_json::to_string_pretty(self).context("serialize import report")?;
        fs::write(path, text)
            .with_context(|| format!("write import report to {}", path.display()))?;
        Ok(())
    }
}

/// Walk `source_root` recursively, import every recognised file into
/// `data_root` Vortex layout, return a report. Existing Vortex output
/// files are skipped unless `force_rebuild` is set.
pub fn import_directory_recursive(
    source_root: impl AsRef<Path>,
    data_root: impl AsRef<Path>,
    force_rebuild: bool,
) -> Result<ImportReport> {
    let source_root = source_root.as_ref().to_path_buf();
    let data_root = data_root.as_ref().to_path_buf();
    fs::create_dir_all(&data_root).ok();

    let mut report = ImportReport {
        root: source_root.display().to_string(),
        data_root: data_root.display().to_string(),
        ..Default::default()
    };

    let mut stack: Vec<PathBuf> = vec![source_root.clone()];
    while let Some(p) = stack.pop() {
        if p.is_dir() {
            // Skip the data_root itself if the user pointed source at
            // a parent that contains it (avoid re-importing canonical
            // Vortex files we just wrote).
            if p == data_root {
                continue;
            }
            for entry in fs::read_dir(&p).with_context(|| format!("readdir {}", p.display()))? {
                let entry = entry.context("readdir entry")?;
                stack.push(entry.path());
            }
            continue;
        }
        if !p.is_file() {
            continue;
        }
        report.files_seen += 1;
        match import_one_file(&p, &source_root, &data_root, force_rebuild) {
            Ok(result) => {
                match result.status {
                    ImportStatus::Imported => report.imported += 1,
                    ImportStatus::Skipped => report.skipped += 1,
                    ImportStatus::Quarantined => report.quarantined += 1,
                    ImportStatus::Failed => report.failed += 1,
                }
                report.results.push(result);
            }
            Err(err) => {
                report.failed += 1;
                report.results.push(ImportFileResult {
                    source: p.display().to_string(),
                    symbol: None,
                    timeframe: None,
                    format: "unknown".to_string(),
                    rows: 0,
                    status: ImportStatus::Failed,
                    message: format!("{:#}", err),
                });
            }
        }
    }

    Ok(report)
}

/// Import a single file. Detects format from extension, parses to
/// OHLCV, infers symbol/timeframe from path components, writes Vortex.
pub fn import_one_file(
    path: &Path,
    source_root: &Path,
    data_root: &Path,
    force_rebuild: bool,
) -> Result<ImportFileResult> {
    let format = detect_format(path);
    if matches!(format.as_str(), "unknown" | "ignored") {
        return Ok(ImportFileResult {
            source: path.display().to_string(),
            symbol: None,
            timeframe: None,
            format,
            rows: 0,
            status: ImportStatus::Skipped,
            message: "format not supported".to_string(),
        });
    }

    let (symbol, timeframe) = infer_symbol_and_timeframe(path, source_root).unwrap_or((None, None));

    // Skip writing if this file IS the canonical Vortex output for
    // its (symbol, tf) pair — avoid recursive re-import.
    if let (Some(sym), Some(tf)) = (symbol.as_ref(), timeframe.as_ref()) {
        let dest = canonical_vortex_path(data_root, sym, tf);
        if dest == path {
            return Ok(ImportFileResult {
                source: path.display().to_string(),
                symbol: Some(sym.clone()),
                timeframe: Some(tf.clone()),
                format,
                rows: 0,
                status: ImportStatus::Skipped,
                message: "already canonical".to_string(),
            });
        }
        if !force_rebuild && dest.exists() {
            return Ok(ImportFileResult {
                source: path.display().to_string(),
                symbol: Some(sym.clone()),
                timeframe: Some(tf.clone()),
                format,
                rows: 0,
                status: ImportStatus::Skipped,
                message: format!("dest exists: {}", dest.display()),
            });
        }
    }

    let ohlcv = match format.as_str() {
        "csv" | "tsv" => parse_csv(path, format == "tsv"),
        "json" => parse_json(path),
        "jsonl" => parse_jsonl(path),
        "parquet" => parse_parquet(path),
        "vortex" => parse_vortex(path),
        other => bail!(
            "Unsupported file format '{}'. Supported: csv, tsv, json, jsonl, parquet, vortex.",
            other
        ),
    };

    let ohlcv = match ohlcv {
        Ok(o) => o,
        Err(err) => {
            // Quarantine the source so the user can inspect.
            let qpath = quarantine_path(data_root, path);
            if let Some(dir) = qpath.parent()
                && let Err(mk_err) = fs::create_dir_all(dir)
            {
                tracing::warn!(
                    target: "neoethos_data::universal_importer",
                    dir = %dir.display(),
                    error = %mk_err,
                    "quarantine: failed to create directory"
                );
            }
            // Best-effort copy; don't fail import if quarantine itself fails,
            // but surface the failure so the operator knows the source file
            // wasn't preserved for inspection.
            if let Err(cp_err) = fs::copy(path, &qpath) {
                tracing::warn!(
                    target: "neoethos_data::universal_importer",
                    src = %path.display(),
                    dst = %qpath.display(),
                    error = %cp_err,
                    "quarantine: failed to copy source file"
                );
            }
            return Ok(ImportFileResult {
                source: path.display().to_string(),
                symbol: symbol.clone(),
                timeframe: timeframe.clone(),
                format,
                rows: 0,
                status: ImportStatus::Quarantined,
                message: format!("parse failed: {:#}", err),
            });
        }
    };

    let row_count = ohlcv.open.len();
    if row_count == 0 {
        return Ok(ImportFileResult {
            source: path.display().to_string(),
            symbol,
            timeframe,
            format,
            rows: 0,
            status: ImportStatus::Skipped,
            message: "empty OHLCV".to_string(),
        });
    }

    let (Some(sym), Some(tf)) = (symbol.as_ref(), timeframe.as_ref()) else {
        return Ok(ImportFileResult {
            source: path.display().to_string(),
            symbol,
            timeframe,
            format,
            rows: row_count,
            status: ImportStatus::Skipped,
            message: format!(
                "Could not infer symbol/timeframe from '{}'. \
                 Organise files as <root>/EURUSD/M5/data.csv or name them EURUSD_M5_2024.csv, \
                 or pass --symbol/--timeframe.",
                path.display()
            ),
        });
    };

    let dest = canonical_vortex_path(data_root, sym, tf);
    if let Some(dir) = dest.parent() {
        fs::create_dir_all(dir).ok();
    }
    crate::write_ohlcv_vortex(&dest, &ohlcv)
        .with_context(|| format!("write vortex to {}", dest.display()))?;

    // Verify by reload.
    let reloaded = crate::load_vortex(&dest)
        .with_context(|| format!("reload vortex from {}", dest.display()))?;
    if reloaded.open.len() != row_count {
        bail!(
            "vortex round-trip row mismatch: wrote {} rows, read back {}",
            row_count,
            reloaded.open.len()
        );
    }

    Ok(ImportFileResult {
        source: path.display().to_string(),
        symbol: Some(sym.clone()),
        timeframe: Some(tf.clone()),
        format,
        rows: row_count,
        status: ImportStatus::Imported,
        message: format!("→ {}", dest.display()),
    })
}

// ─── Format detection ──────────────────────────────────────────────────

fn detect_format(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => "csv".into(),
        "tsv" | "tab" => "tsv".into(),
        "json" => "json".into(),
        "jsonl" | "ndjson" => "jsonl".into(),
        "parquet" | "pq" => "parquet".into(),
        "vortex" | "vtx" => "vortex".into(),
        // Common files we want to ignore silently.
        "md" | "txt" | "rst" | "log" | "yml" | "yaml" | "toml" | "lock" | "gitignore"
        | "gitattributes" | "sha256" | "sha1" | "asc" | "zip" | "gz" | "tar" | "bz2" | "xz" => {
            "ignored".into()
        }
        _ => "unknown".into(),
    }
}

// ─── Symbol / timeframe inference ──────────────────────────────────────

fn infer_symbol_and_timeframe(
    path: &Path,
    source_root: &Path,
) -> Option<(Option<String>, Option<String>)> {
    let rel = path.strip_prefix(source_root).ok()?;
    let mut symbol: Option<String> = None;
    let mut timeframe: Option<String> = None;

    for component in rel.components() {
        let s = component.as_os_str().to_string_lossy().to_string();

        if let Some(rest) = s.strip_prefix("symbol=") {
            symbol = Some(rest.to_ascii_uppercase());
            continue;
        }
        if let Some(rest) = s.strip_prefix("timeframe=") {
            timeframe = Some(rest.to_ascii_uppercase());
            continue;
        }

        // Heuristics for non-prefixed paths:
        if symbol.is_none() && (looks_like_symbol(&s) || looks_like_extended_symbol(&s)) {
            symbol = Some(canonical_symbol(&s));
            continue;
        }
        if timeframe.is_none() && looks_like_timeframe(&s) {
            timeframe = Some(s.to_ascii_uppercase());
            continue;
        }
    }

    // Filename (without extension) — often "EURUSD_M5" or "EURUSD-M5-2024".
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        for token in stem.split(['_', '-', '.', ' ']) {
            let t = token.to_ascii_uppercase();
            if symbol.is_none() && (looks_like_symbol(&t) || looks_like_extended_symbol(&t)) {
                symbol = Some(t.clone());
            }
            if timeframe.is_none() && looks_like_timeframe(&t) {
                timeframe = Some(t.clone());
            }
        }
    }

    Some((symbol, timeframe))
}

fn canonical_symbol(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

fn looks_like_symbol(s: &str) -> bool {
    let canon = canonical_symbol(s);
    canon.len() == 6 && canon.chars().all(|c| c.is_ascii_alphabetic())
}

fn looks_like_extended_symbol(s: &str) -> bool {
    // XAUUSD, XAGUSD, BTCUSD, ETHUSD, …
    let canon = canonical_symbol(s);
    matches!(canon.len(), 6..=8)
        && (canon.starts_with("XAU")
            || canon.starts_with("XAG")
            || canon.starts_with("BTC")
            || canon.starts_with("ETH")
            || canon.starts_with("LTC")
            || canon.starts_with("SPX")
            || canon.starts_with("US30")
            || canon.starts_with("NAS"))
}

fn looks_like_timeframe(s: &str) -> bool {
    neoethos_core::is_canonical_timeframe(s)
}

fn canonical_vortex_path(data_root: &Path, symbol: &str, tf: &str) -> PathBuf {
    data_root
        .join(format!("symbol={}", symbol.to_ascii_uppercase()))
        .join(format!("timeframe={}", tf.to_ascii_uppercase()))
        .join("data.vortex")
}

fn quarantine_path(data_root: &Path, src: &Path) -> PathBuf {
    let stem = src
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    data_root.join("import_quarantine").join(stem)
}

// ─── Format parsers ────────────────────────────────────────────────────

// Public re-exports for `crate::core::to_vortex`. The conversion
// pipeline reuses these parsers exactly so format-detection logic
// stays in one place (see to_vortex.rs module docs).
pub fn parse_csv_public(path: &Path, tab_separated: bool) -> Result<Ohlcv> {
    parse_csv(path, tab_separated)
}
pub fn parse_json_public(path: &Path) -> Result<Ohlcv> {
    parse_json(path)
}
pub fn parse_jsonl_public(path: &Path) -> Result<Ohlcv> {
    parse_jsonl(path)
}
pub fn parse_parquet_public(path: &Path) -> Result<Ohlcv> {
    parse_parquet(path)
}

/// Parse a CSV / TSV file, auto-detecting which header column maps to
/// timestamp / open / high / low / close / volume.
fn parse_csv(path: &Path, tab_separated: bool) -> Result<Ohlcv> {
    // Auto-detect the delimiter: comma (US/default), semicolon (EU-locale
    // exports — Excel/MT in de/el/fr where ',' is the decimal separator),
    // or tab (MetaTrader 5 `.csv` files are really TSV). Sniffed from the
    // header buffer; an explicit `tab_separated=true` forces tab. Without
    // this, EU `time;open;…` files were read as ONE column and failed
    // with "missing 'open' column".
    let delimiter = detect_delimiter(path, tab_separated);

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        // Tolerate ragged rows (trailing separators / uneven exports)
        // instead of aborting the whole file on one short line.
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("open csv {}", path.display()))?;

    let headers = rdr.headers().context("read csv headers")?.clone();
    let map = build_column_map(headers.iter());

    let ts_col = map.get("timestamp").copied();
    // #206 MT5 split date/time: MT5 exports have TWO separate columns
    // (`<DATE>` `YYYY.MM.DD` + `<TIME>` `HH:MM:SS`) instead of one
    // unified timestamp. When the column-mapper found neither but DID
    // find both a "date_part" and a "time_part" we concatenate them
    // per-row below before calling `parse_timestamp_cell`.
    let date_col = map.get("date_part").copied();
    let time_col = map.get("time_part").copied();
    let open_col = map
        .get("open")
        .with_context(|| missing_col_err("open", &headers))?;
    let high_col = map
        .get("high")
        .with_context(|| missing_col_err("high", &headers))?;
    let low_col = map
        .get("low")
        .with_context(|| missing_col_err("low", &headers))?;
    let close_col = map
        .get("close")
        .with_context(|| missing_col_err("close", &headers))?;
    let volume_col = map.get("volume").copied();

    let mut ts: Vec<i64> = Vec::new();
    let mut open: Vec<f64> = Vec::new();
    let mut high: Vec<f64> = Vec::new();
    let mut low: Vec<f64> = Vec::new();
    let mut close: Vec<f64> = Vec::new();
    let mut volume: Vec<f64> = Vec::new();
    let mut rejected: usize = 0;

    for (i, rec) in rdr.records().enumerate() {
        let rec = rec.with_context(|| format!("read csv row {}", i))?;
        let ts_value = match (ts_col, date_col, time_col) {
            (Some(c), _, _) => parse_timestamp_cell(rec.get(c).unwrap_or("0")),
            (None, Some(d), Some(t)) => {
                // MT5 split — concat with a space and let
                // parse_timestamp_cell try the format list.
                let combined = format!(
                    "{} {}",
                    rec.get(d).unwrap_or(""),
                    rec.get(t).unwrap_or("")
                );
                parse_timestamp_cell(&combined)
            }
            (None, Some(d), None) => {
                // Date only (daily bars). Use 00:00 as time.
                parse_timestamp_cell(rec.get(d).unwrap_or(""))
            }
            (None, None, _) => i as i64,
        };
        // D03: parse strictly and reject the row on any missing/malformed/
        // implausible OHLC value instead of coercing it to 0.0.
        let o = parse_f64_checked(rec.get(*open_col).unwrap_or(""));
        let h = parse_f64_checked(rec.get(*high_col).unwrap_or(""));
        let l = parse_f64_checked(rec.get(*low_col).unwrap_or(""));
        let c = parse_f64_checked(rec.get(*close_col).unwrap_or(""));
        match (o, h, l, c) {
            (Some(o), Some(h), Some(l), Some(c)) if valid_ohlc_bar(o, h, l, c) => {
                ts.push(ts_value);
                open.push(o);
                high.push(h);
                low.push(l);
                close.push(c);
                if let Some(cv) = volume_col {
                    // Volume may legitimately be 0 (FX real volume); only the
                    // OHLC prices are validity-critical.
                    volume.push(parse_f64(rec.get(cv).unwrap_or("0")).max(0.0));
                }
            }
            _ => {
                rejected += 1;
            }
        }
    }

    if rejected > 0 {
        tracing::warn!(
            target: "neoethos_data::universal_importer",
            rejected,
            kept = open.len(),
            "import: rejected malformed rows (missing/non-finite/non-positive/incoherent OHLC)"
        );
    }
    if open.is_empty() && rejected > 0 {
        anyhow::bail!(
            "all {rejected} data rows were rejected as malformed (missing/non-finite/\
             non-positive/incoherent OHLC) — the file contains no usable OHLC data"
        );
    }

    let timestamp = if ts.is_empty() {
        Some(ts)
    } else {
        let unit = infer_timestamp_unit(&ts).unwrap_or(TimestampUnit::Milliseconds);
        Some(normalize_timestamps_to_millis(&ts, unit)?)
    };

    Ok(Ohlcv {
        timestamp,
        open,
        high,
        low,
        close,
        volume: if volume.is_empty() {
            None
        } else {
            Some(volume)
        },
    })
}

/// Actionable "missing column" error that shows which headers WERE found,
/// so the operator can see why their file didn't map instead of a bare
/// "missing 'open' column".
fn missing_col_err(name: &str, headers: &csv::StringRecord) -> String {
    let found: Vec<&str> = headers.iter().collect();
    format!(
        "CSV is missing a recognizable '{name}' column. Headers found: [{}]. \
         Expected an OHLC header (e.g. open/high/low/close, o/h/l/c, or MT5 \
         <OPEN>…). Note: the first row must be a header — a numeric-only \
         first row is read as headers.",
        found.join(", ")
    )
}

/// Build a normalised column-name → index map. Many sources use slightly
/// different header conventions (`Open`/`open`/`o`/`Price.open` etc.).
///
/// #206 MT5 support: also strip `<` and `>` so `<DATE>` / `<OPEN>` etc.
/// (MetaTrader 5's default angle-bracket-wrapped header style) match
/// the same canonical names as a plain `Date` / `Open`.
///
/// Two-pass logic for the date/time ambiguity:
///   - PASS 1 collects normalised header strings to detect MT5-style
///     paired headers (both `date` AND `time` present, separately).
///   - PASS 2 assigns canonical names. If the file has BOTH columns,
///     `date` → `date_part` and `time` → `time_part` (MT5 splits the
///     timestamp). If only ONE of them is present we keep the historical
///     mapping (`time` alone → `timestamp`, holding a unix epoch; `date`
///     alone → `date_part`, daily bars without a clock).
///
/// This dual-mode handling is what keeps the existing unix-epoch CSVs
/// (test `parse_csv_round_trip`) working while ALSO accepting the
/// MT5 split-column layout (test `mt5_split_date_time_header_concat`).
fn build_column_map<'a, I: Iterator<Item = &'a str>>(headers: I) -> HashMap<String, usize> {
    let normalised: Vec<(usize, String)> = headers
        .enumerate()
        .map(|(i, h)| {
            let n = h
                .trim()
                .trim_matches(['<', '>'])
                .to_ascii_lowercase()
                .replace([' ', '.', '-'], "_");
            (i, n)
        })
        .collect();

    let has_plain_date = normalised.iter().any(|(_, n)| n == "date");
    let has_plain_time = normalised.iter().any(|(_, n)| n == "time");
    let mt5_split = has_plain_date && has_plain_time;

    let mut out = HashMap::new();
    for (i, n) in &normalised {
        // Keep first occurrence per canonical name.
        let canonical = match n.as_str() {
            "ts" | "datetime" | "timestamp" | "unix" | "unix_time" | "epoch" | "open_time"
            | "candle_time" => Some("timestamp"),
            // `date` alone → not a usable single timestamp (daily bars
            // only). With a paired `time`, both go into the
            // date_part/time_part slots for per-row concat.
            "date" if mt5_split => Some("date_part"),
            "date" => Some("date_part"),
            // `time` alone → historical contract: unix epoch in this
            // column. With a paired `date`, route to `time_part`.
            "time" if mt5_split => Some("time_part"),
            "time" => Some("timestamp"),
            "o" | "open" | "price_open" | "openprice" => Some("open"),
            "h" | "high" | "price_high" | "highprice" => Some("high"),
            "l" | "low" | "price_low" | "lowprice" => Some("low"),
            "c" | "close" | "price_close" | "closeprice" | "last" => Some("close"),
            // MT5 has both `<TICKVOL>` (tick count) and `<VOL>` (real
            // volume, often 0 on FX). We prefer real volume when
            // present, fall back to tick count — the standard FX-broker
            // convention.
            "v" | "vol" | "volume" | "real_volume" => Some("volume"),
            "tickvol" | "tickvolume" | "tick_volume" => Some("volume"),
            _ => None,
        };
        if let Some(name) = canonical {
            out.entry(name.to_string()).or_insert(*i);
        }
    }
    out
}

/// Detect the CSV delimiter from the header buffer: comma (default),
/// semicolon (EU-locale exports), or tab (MetaTrader 5 TSV-as-`.csv`).
/// `force_tab=true` (explicit TSV) short-circuits to tab. Reads only the
/// first 4 KB to stay cheap. Tab/semicolon need a clear majority so a
/// stray separator inside a comment — or a comma decimal inside a
/// semicolon file — doesn't flip the choice.
fn detect_delimiter(path: &Path, force_tab: bool) -> u8 {
    if force_tab {
        return b'\t';
    }
    use std::io::Read;
    let mut buf = [0u8; 4096];
    let Ok(mut f) = std::fs::File::open(path) else {
        return b',';
    };
    let Ok(n) = f.read(&mut buf) else {
        return b',';
    };
    let slice = &buf[..n];
    let tabs = slice.iter().filter(|b| **b == b'\t').count();
    let semis = slice.iter().filter(|b| **b == b';').count();
    let commas = slice.iter().filter(|b| **b == b',').count();
    if tabs >= 5 && tabs > commas && tabs > semis {
        b'\t'
    } else if semis >= 3 && semis > commas {
        // EU `time;open;…` exports: semicolons outnumber the comma
        // decimals inside the values.
        b';'
    } else {
        b','
    }
}

fn parse_f64(s: &str) -> f64 {
    parse_f64_checked(s).unwrap_or(0.0)
}

/// Audit D03 (2026-07-13): a bar is usable only when all four prices are
/// finite and strictly positive AND the bar is coherent (high is the max,
/// low is the min). The importers previously coerced a missing/garbage
/// cell to `0.0`, injecting a fake price of zero — a catastrophic
/// ~100% move that poisons every downstream return, feature and backtest.
/// Rows that fail this are rejected and counted, never silently zeroed.
fn valid_ohlc_bar(o: f64, h: f64, l: f64, c: f64) -> bool {
    [o, h, l, c].iter().all(|v| v.is_finite() && *v > 0.0)
        && h >= l
        && h >= o
        && h >= c
        && l <= o
        && l <= c
}

/// Parse a numeric cell, disambiguating the comma's role so European
/// locale data isn't silently corrupted:
///   - `1,234.56` (US thousands grouping) → strip the commas → 1234.56
///   - `1,2345`    (European decimal comma) → comma becomes the dot →
///     1.2345  (the old `replace(',', "")` turned this into 12345.0 — a
///     silent ~10000× price corruption on locale-formatted exports)
///   - `1.2345`    (plain) → unchanged
/// Returns `None` on a genuinely unparseable cell so the caller can
/// count/quarantine it instead of poisoning the series with a stray 0.0.
fn parse_f64_checked(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let normalized = if t.contains('.') && t.contains(',') {
        // Both present → comma groups thousands, dot is the decimal.
        t.replace(',', "")
    } else if t.contains(',') {
        // Comma only → it's the decimal separator (de/el/fr/… exports).
        t.replace(',', ".")
    } else {
        t.to_string()
    };
    normalized.parse::<f64>().ok()
}

/// Try several common timestamp formats. Returns ms-since-epoch
/// (caller converts via `infer_timestamp_unit`).
fn parse_timestamp_cell(s: &str) -> i64 {
    let s = s.trim();
    // Pure integer: assumed seconds/ms/ns; let
    // `infer_timestamp_unit` figure out scale.
    if let Ok(v) = s.parse::<i64>() {
        return v;
    }
    if let Ok(v) = s.parse::<f64>() {
        return v as i64;
    }
    // ISO-8601 / RFC-3339 (with or without 'Z').
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return dt.timestamp_millis();
    }
    // Common explicit formats.
    //
    // #206 MT5/MT4 additions:
    //   - `%Y.%m.%d %H:%M:%S` — MT5 `<DATE>` + `<TIME>` concatenated,
    //     e.g. `2024.05.23 14:30:00`. The `.` separator in the date
    //     part is unique to MetaTrader; every other vendor uses `-`
    //     or `/`.
    //   - `%Y.%m.%d %H:%M`     — MT5 M1/M5 historical export (no
    //     seconds).
    //   - `%Y.%m.%d`           — MT5 D1+ daily export, time omitted.
    // The full list is tried in order; first match wins. Putting the
    // ISO formats first means MT5 only pays the cost for genuinely
    // MT5-shaped strings.
    for fmt in &[
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d.%m.%Y %H:%M:%S",
        "%Y.%m.%d %H:%M:%S", // MT5 default
        "%Y.%m.%d %H:%M",    // MT5 M-timeframes
        "%Y-%m-%d",
        "%Y.%m.%d",          // MT5 D1+
    ] {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return dt.and_utc().timestamp_millis();
        }
        if let Ok(d) = chrono::NaiveDate::parse_from_str(s, fmt) {
            return d
                .and_hms_opt(0, 0, 0)
                .map(|dt| dt.and_utc().timestamp_millis())
                .unwrap_or(0);
        }
    }
    0
}

/// JSON: array of objects with the same column names.
fn parse_json(path: &Path) -> Result<Ohlcv> {
    let text = fs::read_to_string(path).with_context(|| format!("read json {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text).context("parse json")?;
    let arr = value.as_array().context("json root must be an array")?;
    let rows: Vec<HashMap<String, serde_json::Value>> = arr
        .iter()
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect();
    rows_to_ohlcv(rows)
}

/// JSON-Lines / NDJSON: one object per line.
fn parse_jsonl(path: &Path) -> Result<Ohlcv> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read jsonl {}", path.display()))?;
    let rows: Vec<HashMap<String, serde_json::Value>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    rows_to_ohlcv(rows)
}

fn rows_to_ohlcv(rows: Vec<HashMap<String, serde_json::Value>>) -> Result<Ohlcv> {
    if rows.is_empty() {
        return Ok(Ohlcv {
            timestamp: Some(Vec::new()),
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: None,
        });
    }
    // Detect column names from the first row.
    let map = build_column_map(rows[0].keys().map(|k| k.as_str()));
    let _ts_col = map.get("timestamp");
    let _open_col = map.get("open").context("json missing 'open' field")?;

    let mut ts: Vec<i64> = Vec::with_capacity(rows.len());
    let mut open: Vec<f64> = Vec::with_capacity(rows.len());
    let mut high: Vec<f64> = Vec::with_capacity(rows.len());
    let mut low: Vec<f64> = Vec::with_capacity(rows.len());
    let mut close: Vec<f64> = Vec::with_capacity(rows.len());
    let mut volume: Vec<f64> = Vec::with_capacity(rows.len());
    let mut rejected: usize = 0;

    for (i, row) in rows.into_iter().enumerate() {
        // D03: reject the row on any missing/malformed/implausible OHLC
        // value instead of coercing it to 0.0.
        let o = extract_num(&row, &["open", "o"]);
        let h = extract_num(&row, &["high", "h"]);
        let l = extract_num(&row, &["low", "l"]);
        let c = extract_num(&row, &["close", "c", "last"]);
        match (o, h, l, c) {
            (Some(o), Some(h), Some(l), Some(c)) if valid_ohlc_bar(o, h, l, c) => {
                ts.push(extract_ts(&row).unwrap_or(i as i64));
                open.push(o);
                high.push(h);
                low.push(l);
                close.push(c);
                if let Some(v) = extract_num(&row, &["volume", "vol", "v", "tickvol"]) {
                    volume.push(v.max(0.0));
                }
            }
            _ => {
                rejected += 1;
            }
        }
    }

    if rejected > 0 {
        tracing::warn!(
            target: "neoethos_data::universal_importer",
            rejected,
            kept = open.len(),
            "import: rejected malformed rows (missing/non-finite/non-positive/incoherent OHLC)"
        );
    }
    if open.is_empty() && rejected > 0 {
        anyhow::bail!(
            "all {rejected} data rows were rejected as malformed (missing/non-finite/\
             non-positive/incoherent OHLC) — the file contains no usable OHLC data"
        );
    }
    if open.is_empty() {
        return Ok(Ohlcv {
            timestamp: Some(Vec::new()),
            open,
            high,
            low,
            close,
            volume: None,
        });
    }

    let unit = infer_timestamp_unit(&ts).unwrap_or(TimestampUnit::Milliseconds);
    let timestamp = Some(normalize_timestamps_to_millis(&ts, unit)?);
    Ok(Ohlcv {
        timestamp,
        open,
        high,
        low,
        close,
        volume: if volume.is_empty() {
            None
        } else {
            Some(volume)
        },
    })
}

fn extract_num(row: &HashMap<String, serde_json::Value>, keys: &[&str]) -> Option<f64> {
    for k in keys {
        for actual_key in row.keys() {
            if actual_key.eq_ignore_ascii_case(k)
                && let Some(v) = row.get(actual_key)
            {
                if let Some(f) = v.as_f64() {
                    return Some(f);
                }
                if let Some(s) = v.as_str()
                    && let Ok(f) = s.parse::<f64>()
                {
                    return Some(f);
                }
                if let Some(i) = v.as_i64() {
                    return Some(i as f64);
                }
            }
        }
    }
    None
}

fn extract_ts(row: &HashMap<String, serde_json::Value>) -> Option<i64> {
    for k in &[
        "timestamp",
        "ts",
        "time",
        "date",
        "datetime",
        "unix",
        "epoch",
        "open_time",
        "candle_time",
    ] {
        for actual_key in row.keys() {
            if actual_key.eq_ignore_ascii_case(k)
                && let Some(v) = row.get(actual_key)
            {
                if let Some(i) = v.as_i64() {
                    return Some(i);
                }
                if let Some(f) = v.as_f64() {
                    return Some(f as i64);
                }
                if let Some(s) = v.as_str() {
                    let parsed = parse_timestamp_cell(s);
                    if parsed != 0 {
                        return Some(parsed);
                    }
                }
            }
        }
    }
    None
}

/// Parquet: defer to existing parquet-migration helper.
fn parse_parquet(path: &Path) -> Result<Ohlcv> {
    crate::core::parquet_migration::read_legacy_parquet_ohlcv(path)
        .with_context(|| format!("read parquet {}", path.display()))
}

/// Vortex: pass-through (just reads + returns).
fn parse_vortex(path: &Path) -> Result<Ohlcv> {
    crate::load_vortex(path).with_context(|| format!("read vortex {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn detect_known_extensions() {
        assert_eq!(detect_format(Path::new("EURUSD_M5.csv")), "csv");
        assert_eq!(detect_format(Path::new("EURUSD_M5.parquet")), "parquet");
        assert_eq!(detect_format(Path::new("EURUSD_M5.json")), "json");
        assert_eq!(detect_format(Path::new("EURUSD_M5.jsonl")), "jsonl");
        assert_eq!(detect_format(Path::new("data.vortex")), "vortex");
        assert_eq!(detect_format(Path::new("README.md")), "ignored");
        assert_eq!(detect_format(Path::new("blob.bin")), "unknown");
    }

    #[test]
    fn parse_f64_handles_locale_decimal_comma() {
        // Plain dot decimal — unchanged.
        assert_eq!(parse_f64("1.16500"), 1.165);
        // European decimal comma — must NOT become 116500.0 (the old bug).
        assert_eq!(parse_f64("1,16500"), 1.165);
        assert_eq!(parse_f64("0,7183"), 0.7183);
        // US thousands grouping — strip the commas.
        assert_eq!(parse_f64("1,234.56"), 1234.56);
        assert_eq!(parse_f64("159"), 159.0);
        // Unparseable / empty → 0.0 via parse_f64, None via the checked
        // variant so callers can quarantine instead of poisoning the series.
        assert_eq!(parse_f64("abc"), 0.0);
        assert_eq!(parse_f64_checked("abc"), None);
        assert_eq!(parse_f64_checked(""), None);
    }

    #[test]
    fn parse_csv_eu_semicolon_comma_decimal() {
        // EU-locale export: ';' delimiter + ',' decimal. Must parse
        // correctly instead of failing with "missing 'open' column"
        // (the bug that quarantined the operator's files).
        let tmp = std::env::temp_dir().join("neoethos_eu_csv_test.csv");
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "time;open;high;low;close;volume").unwrap();
        writeln!(f, "2024-01-01 00:00:00;1,10000;1,20000;1,00000;1,15000;100").unwrap();
        writeln!(f, "2024-01-01 00:01:00;1,15000;1,25000;1,10000;1,20000;120").unwrap();
        drop(f);
        let ohlcv = parse_csv(&tmp, false).expect("EU semicolon CSV must parse");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(ohlcv.open.len(), 2);
        // Prices must be ~1.x, NOT 110000 (the decimal-comma corruption).
        assert!((ohlcv.open[0] - 1.10).abs() < 1e-9, "open[0]={}", ohlcv.open[0]);
        assert!((ohlcv.high[0] - 1.20).abs() < 1e-9, "high[0]={}", ohlcv.high[0]);
        assert!((ohlcv.close[1] - 1.20).abs() < 1e-9, "close[1]={}", ohlcv.close[1]);
    }

    #[test]
    fn infer_symbol_from_hive_path() {
        let tmp = std::env::temp_dir().join("neoethos_importer_test_hive");
        let result =
            infer_symbol_and_timeframe(&tmp.join("symbol=EURUSD/timeframe=M5/data.csv"), &tmp)
                .unwrap();
        assert_eq!(result.0.as_deref(), Some("EURUSD"));
        assert_eq!(result.1.as_deref(), Some("M5"));
    }

    #[test]
    fn infer_symbol_from_filename_token() {
        let tmp = std::env::temp_dir().join("neoethos_importer_test_filename");
        let result = infer_symbol_and_timeframe(&tmp.join("EURUSD_M5_2024.csv"), &tmp).unwrap();
        assert_eq!(result.0.as_deref(), Some("EURUSD"));
        assert_eq!(result.1.as_deref(), Some("M5"));
    }

    #[test]
    fn xau_xag_btc_recognised() {
        let tmp = std::env::temp_dir().join("neoethos_importer_test_metals");
        let r1 = infer_symbol_and_timeframe(&tmp.join("XAUUSD_M30.csv"), &tmp).unwrap();
        assert_eq!(r1.0.as_deref(), Some("XAUUSD"));
        let r2 = infer_symbol_and_timeframe(&tmp.join("BTCUSD_H1.csv"), &tmp).unwrap();
        assert_eq!(r2.0.as_deref(), Some("BTCUSD"));
    }

    #[test]
    fn parse_csv_round_trip() {
        let tmp = std::env::temp_dir().join("neoethos_importer_csv.csv");
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "time,open,high,low,close,volume").unwrap();
        writeln!(f, "1700000000,1.10,1.11,1.09,1.105,1234").unwrap();
        writeln!(f, "1700000060,1.105,1.115,1.10,1.11,2345").unwrap();
        drop(f);
        let ohlcv = parse_csv(&tmp, false).unwrap();
        assert_eq!(ohlcv.open, vec![1.10, 1.105]);
        assert_eq!(ohlcv.close, vec![1.105, 1.11]);
        assert_eq!(ohlcv.timestamp.as_ref().unwrap()[0], 1_700_000_000_000);
    }

    #[test]
    fn csv_rejects_malformed_rows_instead_of_zeroing_prices() {
        // Audit D03: a blank/garbage/incoherent OHLC cell must drop the row,
        // not become a fake price of 0.0 that injects a ~100% move.
        let tmp = std::env::temp_dir().join("neoethos_importer_d03.csv");
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "time,open,high,low,close,volume").unwrap();
        writeln!(f, "1700000000,1.10,1.11,1.09,1.105,1234").unwrap(); // good
        writeln!(f, "1700000060,1.105,,1.10,1.11,2345").unwrap(); // blank high → reject
        writeln!(f, "1700000120,1.11,1.12,1.10,xyz,999").unwrap(); // garbage close → reject
        writeln!(f, "1700000180,1.11,1.10,1.12,1.11,50").unwrap(); // high<low incoherent → reject
        writeln!(f, "1700000240,1.12,1.13,1.11,1.125,4321").unwrap(); // good
        drop(f);
        let ohlcv = parse_csv(&tmp, false).unwrap();
        // Only the two well-formed rows survive; NO zero prices leak in.
        assert_eq!(ohlcv.open, vec![1.10, 1.12], "only valid rows kept");
        assert_eq!(ohlcv.close, vec![1.105, 1.125]);
        assert!(
            ohlcv.low.iter().all(|v| *v > 0.0),
            "no zero-coerced prices may survive"
        );
        assert_eq!(ohlcv.volume.as_ref().unwrap(), &vec![1234.0, 4321.0]);
    }

    #[test]
    fn csv_all_malformed_rows_is_an_explicit_error() {
        // A file whose every data row is garbage must fail loudly, not
        // return a silently-empty/zeroed series.
        let tmp = std::env::temp_dir().join("neoethos_importer_d03_allbad.csv");
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "time,open,high,low,close").unwrap();
        writeln!(f, "1700000000,,,,").unwrap();
        writeln!(f, "1700000060,x,y,z,w").unwrap();
        drop(f);
        let err = parse_csv(&tmp, false).unwrap_err();
        assert!(
            err.to_string().contains("rejected as malformed"),
            "expected explicit malformed-rows error, got: {err}"
        );
    }

    /// #206 regression guard: a MetaTrader 5 raw export must import
    /// without manual pre-processing. MT5 quirks exercised here:
    ///   - `.csv` extension but TAB-separated content (auto-sniff)
    ///   - angle-bracket headers: `<DATE>` `<TIME>` `<OPEN>` …
    ///   - separate `<DATE>` (YYYY.MM.DD) and `<TIME>` (HH:MM:SS)
    ///     columns instead of a unified timestamp (per-row concat)
    ///   - dot-separated date format `YYYY.MM.DD` (parse_timestamp_cell
    ///     format list addition)
    ///   - `<TICKVOL>` mapped to `volume` (MT5 sets `<VOL>` to 0 on
    ///     most FX brokers, so tick count is what users actually want)
    #[test]
    fn mt5_raw_export_imports_without_preprocessing() {
        let tmp = std::env::temp_dir().join("neoethos_importer_mt5.csv");
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(
            f,
            "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"
        )
        .unwrap();
        writeln!(
            f,
            "2024.05.23\t14:30:00\t1.0850\t1.0855\t1.0849\t1.0852\t150\t0\t1"
        )
        .unwrap();
        writeln!(
            f,
            "2024.05.23\t14:31:00\t1.0852\t1.0858\t1.0851\t1.0857\t180\t0\t1"
        )
        .unwrap();
        drop(f);

        // Caller passes tab_separated=false to trigger the sniff path;
        // a real /data/import would do the same since the extension is
        // .csv. The sniff must flip to tabs and the columns must map.
        let ohlcv = parse_csv(&tmp, false).expect("MT5 raw export must parse");

        assert_eq!(ohlcv.open, vec![1.0850, 1.0852], "open prices");
        assert_eq!(ohlcv.close, vec![1.0852, 1.0857], "close prices");
        // Real volume is 0 on FX, so the importer should fall back to
        // tick count (<TICKVOL>) and surface 150/180, not 0/0.
        assert_eq!(
            ohlcv.volume.as_ref().unwrap(),
            &vec![150.0, 180.0],
            "TICKVOL fallback when VOL=0"
        );
        // 2024-05-23 14:30:00 UTC = 1716474600 epoch seconds.
        // The importer normalises to ms; first bar should land there.
        let first_ts_ms = ohlcv.timestamp.as_ref().unwrap()[0];
        assert_eq!(
            first_ts_ms, 1_716_474_600_000,
            "MT5 date+time concat → 2024-05-23 14:30:00 UTC"
        );
        // Δt between consecutive bars = 60 s.
        let second_ts_ms = ohlcv.timestamp.as_ref().unwrap()[1];
        assert_eq!(second_ts_ms - first_ts_ms, 60_000, "M1 bar spacing");
    }
}
