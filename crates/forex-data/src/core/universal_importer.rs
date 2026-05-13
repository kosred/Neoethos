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
use crate::core::timestamps::{TimestampUnit, infer_timestamp_unit, normalize_timestamps_to_millis};

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
        let text = serde_json::to_string_pretty(self)
            .context("serialize import report")?;
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

    let (symbol, timeframe) = infer_symbol_and_timeframe(path, source_root)
        .unwrap_or((None, None));

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
        other => bail!("internal: detected unsupported format {}", other),
    };

    let ohlcv = match ohlcv {
        Ok(o) => o,
        Err(err) => {
            // Quarantine the source so the user can inspect.
            let qpath = quarantine_path(data_root, path);
            if let Some(dir) = qpath.parent() {
                fs::create_dir_all(dir).ok();
            }
            // Best-effort copy; don't fail import if quarantine itself fails.
            let _ = fs::copy(path, &qpath);
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
            message: "could not infer symbol/timeframe from path".to_string(),
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
        | "gitattributes" | "sha256" | "sha1" | "asc" | "zip" | "gz" | "tar" | "bz2"
        | "xz" => "ignored".into(),
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
        if symbol.is_none()
            && (looks_like_symbol(&s) || looks_like_extended_symbol(&s))
        {
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
    let s = s.to_ascii_uppercase();
    matches!(
        s.as_str(),
        "M1" | "M2"
            | "M3"
            | "M4"
            | "M5"
            | "M6"
            | "M10"
            | "M12"
            | "M15"
            | "M20"
            | "M30"
            | "H1"
            | "H2"
            | "H3"
            | "H4"
            | "H6"
            | "H8"
            | "H12"
            | "D1"
            | "W1"
            | "MN1"
    )
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

/// Parse a CSV / TSV file, auto-detecting which header column maps to
/// timestamp / open / high / low / close / volume.
fn parse_csv(path: &Path, tab_separated: bool) -> Result<Ohlcv> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if tab_separated { b'\t' } else { b',' })
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("open csv {}", path.display()))?;

    let headers = rdr.headers().context("read csv headers")?.clone();
    let map = build_column_map(headers.iter());

    let ts_col = map.get("timestamp").copied();
    let open_col = map.get("open").context("missing 'open' column")?;
    let high_col = map.get("high").context("missing 'high' column")?;
    let low_col = map.get("low").context("missing 'low' column")?;
    let close_col = map.get("close").context("missing 'close' column")?;
    let volume_col = map.get("volume").copied();

    let mut ts: Vec<i64> = Vec::new();
    let mut open: Vec<f64> = Vec::new();
    let mut high: Vec<f64> = Vec::new();
    let mut low: Vec<f64> = Vec::new();
    let mut close: Vec<f64> = Vec::new();
    let mut volume: Vec<f64> = Vec::new();

    for (i, rec) in rdr.records().enumerate() {
        let rec = rec.with_context(|| format!("read csv row {}", i))?;
        let ts_value = match ts_col {
            Some(c) => parse_timestamp_cell(rec.get(c).unwrap_or("0")),
            None => i as i64,
        };
        ts.push(ts_value);
        open.push(parse_f64(rec.get(*open_col).unwrap_or("0")));
        high.push(parse_f64(rec.get(*high_col).unwrap_or("0")));
        low.push(parse_f64(rec.get(*low_col).unwrap_or("0")));
        close.push(parse_f64(rec.get(*close_col).unwrap_or("0")));
        if let Some(c) = volume_col {
            volume.push(parse_f64(rec.get(c).unwrap_or("0")));
        }
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
        volume: if volume.is_empty() { None } else { Some(volume) },
    })
}

/// Build a normalised column-name → index map. Many sources use slightly
/// different header conventions (`Open`/`open`/`o`/`Price.open` etc.).
fn build_column_map<'a, I: Iterator<Item = &'a str>>(headers: I) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for (i, h) in headers.enumerate() {
        let normalised = h
            .trim()
            .to_ascii_lowercase()
            .replace([' ', '.', '-'], "_");
        // Keep first occurrence per canonical name.
        let canonical = match normalised.as_str() {
            "ts" | "time" | "date" | "datetime" | "timestamp" | "unix" | "unix_time"
            | "epoch" | "open_time" | "candle_time" => Some("timestamp"),
            "o" | "open" | "price_open" | "openprice" => Some("open"),
            "h" | "high" | "price_high" | "highprice" => Some("high"),
            "l" | "low" | "price_low" | "lowprice" => Some("low"),
            "c" | "close" | "price_close" | "closeprice" | "last" => Some("close"),
            "v" | "vol" | "volume" | "tickvol" | "tickvolume" | "tick_volume" => {
                Some("volume")
            }
            _ => None,
        };
        if let Some(name) = canonical {
            out.entry(name.to_string()).or_insert(i);
        }
    }
    out
}

fn parse_f64(s: &str) -> f64 {
    s.trim().replace(',', "").parse::<f64>().unwrap_or(0.0)
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
    for fmt in &[
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d.%m.%Y %H:%M:%S",
        "%Y-%m-%d",
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
    let text = fs::read_to_string(path)
        .with_context(|| format!("read json {}", path.display()))?;
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
    let text = fs::read_to_string(path)
        .with_context(|| format!("read jsonl {}", path.display()))?;
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

    for (i, row) in rows.into_iter().enumerate() {
        ts.push(extract_ts(&row).unwrap_or(i as i64));
        open.push(extract_num(&row, &["open", "o"]).unwrap_or(0.0));
        high.push(extract_num(&row, &["high", "h"]).unwrap_or(0.0));
        low.push(extract_num(&row, &["low", "l"]).unwrap_or(0.0));
        close.push(extract_num(&row, &["close", "c", "last"]).unwrap_or(0.0));
        if let Some(v) = extract_num(&row, &["volume", "vol", "v", "tickvol"]) {
            volume.push(v);
        }
    }

    let unit = infer_timestamp_unit(&ts).unwrap_or(TimestampUnit::Milliseconds);
    let timestamp = Some(normalize_timestamps_to_millis(&ts, unit)?);
    Ok(Ohlcv {
        timestamp,
        open,
        high,
        low,
        close,
        volume: if volume.is_empty() { None } else { Some(volume) },
    })
}

fn extract_num(row: &HashMap<String, serde_json::Value>, keys: &[&str]) -> Option<f64> {
    for k in keys {
        for actual_key in row.keys() {
            if actual_key.eq_ignore_ascii_case(k) {
                if let Some(v) = row.get(actual_key) {
                    if let Some(f) = v.as_f64() {
                        return Some(f);
                    }
                    if let Some(s) = v.as_str() {
                        if let Ok(f) = s.parse::<f64>() {
                            return Some(f);
                        }
                    }
                    if let Some(i) = v.as_i64() {
                        return Some(i as f64);
                    }
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
            if actual_key.eq_ignore_ascii_case(k) {
                if let Some(v) = row.get(actual_key) {
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
        assert_eq!(
            detect_format(Path::new("EURUSD_M5.csv")),
            "csv"
        );
        assert_eq!(
            detect_format(Path::new("EURUSD_M5.parquet")),
            "parquet"
        );
        assert_eq!(
            detect_format(Path::new("EURUSD_M5.json")),
            "json"
        );
        assert_eq!(
            detect_format(Path::new("EURUSD_M5.jsonl")),
            "jsonl"
        );
        assert_eq!(
            detect_format(Path::new("data.vortex")),
            "vortex"
        );
        assert_eq!(detect_format(Path::new("README.md")), "ignored");
        assert_eq!(detect_format(Path::new("blob.bin")), "unknown");
    }

    #[test]
    fn infer_symbol_from_hive_path() {
        let tmp = std::env::temp_dir().join("forex_ai_importer_test_hive");
        let result = infer_symbol_and_timeframe(
            &tmp.join("symbol=EURUSD/timeframe=M5/data.csv"),
            &tmp,
        )
        .unwrap();
        assert_eq!(result.0.as_deref(), Some("EURUSD"));
        assert_eq!(result.1.as_deref(), Some("M5"));
    }

    #[test]
    fn infer_symbol_from_filename_token() {
        let tmp = std::env::temp_dir().join("forex_ai_importer_test_filename");
        let result =
            infer_symbol_and_timeframe(&tmp.join("EURUSD_M5_2024.csv"), &tmp).unwrap();
        assert_eq!(result.0.as_deref(), Some("EURUSD"));
        assert_eq!(result.1.as_deref(), Some("M5"));
    }

    #[test]
    fn xau_xag_btc_recognised() {
        let tmp = std::env::temp_dir().join("forex_ai_importer_test_metals");
        let r1 = infer_symbol_and_timeframe(&tmp.join("XAUUSD_M30.csv"), &tmp).unwrap();
        assert_eq!(r1.0.as_deref(), Some("XAUUSD"));
        let r2 = infer_symbol_and_timeframe(&tmp.join("BTCUSD_H1.csv"), &tmp).unwrap();
        assert_eq!(r2.0.as_deref(), Some("BTCUSD"));
    }

    #[test]
    fn parse_csv_round_trip() {
        let tmp = std::env::temp_dir().join("forex_ai_importer_csv.csv");
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "time,open,high,low,close,volume").unwrap();
        writeln!(f, "1700000000,1.10,1.11,1.09,1.105,1234").unwrap();
        writeln!(f, "1700000060,1.105,1.115,1.10,1.11,2345").unwrap();
        drop(f);
        let ohlcv = parse_csv(&tmp, false).unwrap();
        assert_eq!(ohlcv.open, vec![1.10, 1.105]);
        assert_eq!(ohlcv.close, vec![1.105, 1.11]);
        assert_eq!(
            ohlcv.timestamp.as_ref().unwrap()[0],
            1_700_000_000_000
        );
    }
}
